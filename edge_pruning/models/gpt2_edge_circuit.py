"""
Edge-level circuit pruning for GPT-2.

Post-hoc edge pruning on nodes discovered via node-level circuit pruning.
Edges connect source nodes (attention heads, MLPs, embedding) to receiver
nodes (attention head Q/K/V inputs, MLP inputs, output).

Uses the same dual-stream (clean/corrupted) approach as node pruning:
when an edge is pruned (gate=0), the receiver gets the corrupted source's
contribution instead of the clean one for that specific edge.

Reference: Bhaskar et al. — edge-level circuit discovery.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import OrderedDict

from transformers import GPT2LMHeadModel
from .l0 import HardConcreteGate


@dataclass
class EdgePruningConfig:
    lambda_edges: float = 1.0
    sparsity_warmup_steps: int = 500
    include_output_edges: bool = True


class _LogitsOutput:
    """Minimal output wrapper with .logits attribute for evaluation compatibility."""
    def __init__(self, logits):
        self.logits = logits


class EdgePrunableGPT2(nn.Module):
    """
    Edge-level pruning for GPT-2 on surviving nodes from node pruning.

    Nodes:
      - Embedding (source only)
      - Attention heads (source + receiver)
      - MLP blocks (source + receiver)

    Edges (with separate Q/K/V gates for attention receivers):
      - source -> receiver_head Q
      - source -> receiver_head K
      - source -> receiver_head V
      - source -> receiver_MLP
      - source -> output (optional)
    """

    def __init__(
        self,
        base_model: GPT2LMHeadModel,
        active_heads: Dict[int, List[int]],
        active_mlps: Set[int],
        config: EdgePruningConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.edge_config = config
        self.gpt_config = base_model.config

        self.active_heads = active_heads
        self.active_mlps = set(active_mlps)

        self.num_layers = self.gpt_config.n_layer
        self.total_heads = self.gpt_config.n_head
        self.hidden_size = self.gpt_config.hidden_size
        self.head_dim = self.hidden_size // self.total_heads

        self._build_source_index()
        self._create_edge_gates()

    @classmethod
    def from_pretrained_with_edges(
        cls,
        model_name: str,
        active_heads: Dict[int, List[int]],
        active_mlps: Set[int],
        config: EdgePruningConfig,
        **kwargs,
    ):
        base = GPT2LMHeadModel.from_pretrained(model_name, **kwargs)
        for p in base.parameters():
            p.requires_grad = False
        return cls(base, active_heads, active_mlps, config)

    @property
    def config(self):
        return self.gpt_config

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    # ------------------------------------------------------------------
    # Source index
    # ------------------------------------------------------------------

    def _build_source_index(self):
        """Build ordered source list and precompute per-receiver source indices."""
        self.sources = []            # [(type, layer, head_idx)]
        self.source_to_idx = {}      # (type, layer[, head]) -> int

        # Source 0: embedding
        self.sources.append(("emb", -1, -1))
        self.source_to_idx[("emb", -1)] = 0

        for l in range(self.num_layers):
            for h in sorted(self.active_heads.get(l, [])):
                idx = len(self.sources)
                self.sources.append(("head", l, h))
                self.source_to_idx[("head", l, h)] = idx
            if l in self.active_mlps:
                idx = len(self.sources)
                self.sources.append(("mlp", l, -1))
                self.source_to_idx[("mlp", l)] = idx

        self.num_sources = len(self.sources)

        # Sources available to attention at layer l (everything before layer l)
        self.src_idx_before_layer = {}
        # Sources available to MLP at layer l (before + attn heads at layer l)
        self.src_idx_through_attn = {}

        for l in range(self.num_layers):
            self.src_idx_before_layer[l] = [
                i for i, (t, sl, _) in enumerate(self.sources) if sl < l
            ]
            self.src_idx_through_attn[l] = [
                i
                for i, (t, sl, _) in enumerate(self.sources)
                if sl < l or (sl == l and t == "head")
            ]

    # ------------------------------------------------------------------
    # Edge gates
    # ------------------------------------------------------------------

    def _create_edge_gates(self):
        self.attn_q_gates = nn.ModuleDict()
        self.attn_k_gates = nn.ModuleDict()
        self.attn_v_gates = nn.ModuleDict()
        self.mlp_edge_gates = nn.ModuleDict()

        total_edges = 0

        for l in range(self.num_layers):
            n_src = len(self.src_idx_before_layer[l])
            for h in self.active_heads.get(l, []):
                if n_src > 0:
                    key = f"L{l}_H{h}"
                    self.attn_q_gates[key] = HardConcreteGate(n_src)
                    self.attn_k_gates[key] = HardConcreteGate(n_src)
                    self.attn_v_gates[key] = HardConcreteGate(n_src)
                    total_edges += 3 * n_src

            if l in self.active_mlps:
                n_src_mlp = len(self.src_idx_through_attn[l])
                if n_src_mlp > 0:
                    self.mlp_edge_gates[f"L{l}"] = HardConcreteGate(n_src_mlp)
                    total_edges += n_src_mlp

        if self.edge_config.include_output_edges:
            self.output_edge_gates = HardConcreteGate(self.num_sources)
            total_edges += self.num_sources
        else:
            self.output_edge_gates = None

        self.total_edges = total_edges
        print(f"Edge pruning: {total_edges} edge gates across {self.num_sources} source nodes")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        corrupted_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if corrupted_input_ids is None:
            return self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        return self._edge_forward(input_ids, corrupted_input_ids, attention_mask)

    def _edge_forward(self, input_ids, corrupted_input_ids, attention_mask):
        """Dual-stream edge-gated forward pass."""
        transformer = self.base_model.transformer
        B, S = input_ids.shape
        device = input_ids.device
        HD = self.head_dim
        NH = self.total_heads
        H = self.hidden_size

        # -- Embeddings --
        position_ids = torch.arange(S, device=device).unsqueeze(0)
        clean_emb = transformer.drop(
            transformer.wte(input_ids) + transformer.wpe(position_ids)
        )
        corr_emb = transformer.drop(
            transformer.wte(corrupted_input_ids) + transformer.wpe(position_ids)
        ).detach()

        # -- Masks --
        causal = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
        if attention_mask is not None:
            pad = attention_mask.bool().unsqueeze(1)        # [B, 1, S]
            per_head_mask = causal.unsqueeze(0) & pad       # [B, S, S]
            multi_head_mask = per_head_mask.unsqueeze(1)    # [B, 1, S, S]
        else:
            per_head_mask = causal                          # [S, S]
            multi_head_mask = causal[None, None, :, :]      # [1, 1, S, S]

        # -- Deltas: delta[i] = clean_source[i] - corrupted_source[i] --
        deltas = [None] * self.num_sources
        deltas[0] = clean_emb - corr_emb

        corrupted_residual = corr_emb

        # -- Layer-by-layer processing --
        for l in range(self.num_layers):
            block = transformer.h[l]
            active_heads_l = self.active_heads.get(l, [])

            # ============================================================
            # CORRUPTED ATTENTION (all heads, needed for corrupted residual)
            # ============================================================
            corr_ln1 = block.ln_1(corrupted_residual)
            corr_qkv = block.attn.c_attn(corr_ln1)
            cq, ck, cv = corr_qkv.split(H, dim=-1)
            cq = cq.view(B, S, NH, HD).transpose(1, 2)
            ck = ck.view(B, S, NH, HD).transpose(1, 2)
            cv = cv.view(B, S, NH, HD).transpose(1, 2)

            c_scores = (cq @ ck.transpose(-2, -1)) / math.sqrt(HD)
            c_scores = c_scores.masked_fill(~multi_head_mask, float("-inf"))
            c_probs = F.softmax(c_scores, dim=-1)
            c_attn_out = c_probs @ cv                       # [B, NH, S, HD]

            # Full corrupted attention output (including c_proj bias)
            c_attn_flat = c_attn_out.transpose(1, 2).reshape(B, S, H)
            c_attn_output = (
                c_attn_flat @ block.attn.c_proj.weight + block.attn.c_proj.bias
            )

            # ============================================================
            # CLEAN ATTENTION (only active heads, edge-gated)
            # ============================================================
            src_before = self.src_idx_before_layer[l]
            if src_before and active_heads_l:
                delta_stack = torch.stack([deltas[s] for s in src_before])

            for h in active_heads_l:
                key = f"L{l}_H{h}"

                if src_before:
                    qg = self.attn_q_gates[key]().view(-1, 1, 1, 1)
                    kg = self.attn_k_gates[key]().view(-1, 1, 1, 1)
                    vg = self.attn_v_gates[key]().view(-1, 1, 1, 1)

                    q_in = corrupted_residual + (qg * delta_stack).sum(0)
                    k_in = corrupted_residual + (kg * delta_stack).sum(0)
                    v_in = corrupted_residual + (vg * delta_stack).sum(0)
                else:
                    q_in = k_in = v_in = corrupted_residual

                # Per-head Q / K / V via sliced c_attn weights
                q_ln = block.ln_1(q_in)
                k_ln = block.ln_1(k_in)
                v_ln = block.ln_1(v_in)

                w_q = block.attn.c_attn.weight[:, h * HD : (h + 1) * HD]
                b_q = block.attn.c_attn.bias[h * HD : (h + 1) * HD]
                w_k = block.attn.c_attn.weight[:, H + h * HD : H + (h + 1) * HD]
                b_k = block.attn.c_attn.bias[H + h * HD : H + (h + 1) * HD]
                w_v = block.attn.c_attn.weight[:, 2 * H + h * HD : 2 * H + (h + 1) * HD]
                b_v = block.attn.c_attn.bias[2 * H + h * HD : 2 * H + (h + 1) * HD]

                Q = q_ln @ w_q + b_q               # [B, S, HD]
                K = k_ln @ w_k + b_k
                V = v_ln @ w_v + b_v

                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(HD)
                scores = scores.masked_fill(~per_head_mask, float("-inf"))
                probs = F.softmax(scores, dim=-1)
                attn_out_h = probs @ V              # [B, S, HD]

                # Output projection slice (no bias — bias is in corrupted_residual)
                w_o = block.attn.c_proj.weight[h * HD : (h + 1) * HD, :]
                clean_h = attn_out_h @ w_o          # [B, S, H]

                # Corrupted per-head output (same slice, no bias)
                corr_h = c_attn_out[:, h, :, :] @ w_o

                src_idx = self.source_to_idx[("head", l, h)]
                deltas[src_idx] = clean_h - corr_h

            # Update corrupted residual with full attention output
            corrupted_residual = corrupted_residual + c_attn_output

            # ============================================================
            # CORRUPTED MLP (full, needed for corrupted residual)
            # ============================================================
            corr_ln2 = block.ln_2(corrupted_residual)
            corr_mlp_out = block.mlp(corr_ln2)

            # ============================================================
            # CLEAN MLP (if active, edge-gated)
            # ============================================================
            if l in self.active_mlps:
                src_through = self.src_idx_through_attn[l]
                mlp_key = f"L{l}"

                if src_through:
                    mlp_g = self.mlp_edge_gates[mlp_key]().view(-1, 1, 1, 1)
                    ds = torch.stack([deltas[s] for s in src_through])
                    mlp_in = corrupted_residual + (mlp_g * ds).sum(0)
                else:
                    mlp_in = corrupted_residual

                clean_mlp_out = block.mlp(block.ln_2(mlp_in))

                src_idx = self.source_to_idx[("mlp", l)]
                deltas[src_idx] = clean_mlp_out - corr_mlp_out

            corrupted_residual = corrupted_residual + corr_mlp_out

        # ================================================================
        # OUTPUT
        # ================================================================
        all_deltas = torch.stack(
            [d if d is not None else torch.zeros(B, S, H, device=device) for d in deltas]
        )

        if self.output_edge_gates is not None:
            og = self.output_edge_gates().view(-1, 1, 1, 1)
            final_residual = corrupted_residual + (og * all_deltas).sum(0)
        else:
            final_residual = corrupted_residual + all_deltas.sum(0)

        logits = self.base_model.lm_head(transformer.ln_f(final_residual))
        return _LogitsOutput(logits)

    # ------------------------------------------------------------------
    # Sparsity loss
    # ------------------------------------------------------------------

    def get_sparsity_loss(self, step: int = 0):
        warmup = min(
            1.0,
            step / self.edge_config.sparsity_warmup_steps
            if self.edge_config.sparsity_warmup_steps > 0
            else 1.0,
        )

        total, count = torch.tensor(0.0, device=self.device), 0

        for gate_dict in [
            self.attn_q_gates,
            self.attn_k_gates,
            self.attn_v_gates,
            self.mlp_edge_gates,
        ]:
            for gate in gate_dict.values():
                total = total + gate.get_sparsity_loss()
                count += 1

        if self.output_edge_gates is not None:
            total = total + self.output_edge_gates.get_sparsity_loss()
            count += 1

        if count > 0:
            total = self.edge_config.lambda_edges * warmup * total / count

        return {"total_sparsity": total}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_final_circuit_mode(self, enabled: bool):
        cnt = 0
        for m in self.modules():
            if isinstance(m, HardConcreteGate):
                m.final_mode = enabled
                cnt += 1
        print(f"Edge circuit final mode = {enabled} ({cnt} gates)")

    def get_edge_count(self):
        """Return (total_edges, active_edges) with final-mode gates."""
        total, active = 0, 0
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, HardConcreteGate):
                    mask = m()
                    total += mask.numel()
                    active += (mask > 0.5).sum().item()
        return total, int(active)
