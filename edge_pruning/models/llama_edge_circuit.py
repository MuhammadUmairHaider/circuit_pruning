"""
Edge-level circuit pruning for Llama models.

Post-hoc edge pruning on nodes discovered via node-level circuit pruning.
Edges connect source nodes (attention heads, MLPs, embedding) to receiver
nodes (attention head Q/K/V inputs, MLP inputs, output).

Uses the same dual-stream (clean/corrupted) approach as node pruning:
when an edge is pruned (gate=0), the receiver gets the corrupted source's
contribution instead of the clean one for that specific edge.

Handles Llama-specific features:
- GQA (Grouped Query Attention)
- RoPE (Rotary Positional Embeddings)
- SwiGLU MLP
- RMSNorm (instead of LayerNorm)
- Separate q_proj, k_proj, v_proj, o_proj (instead of c_attn / c_proj)
- No bias in projections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from transformers import LlamaForCausalLM
from .l0 import HardConcreteGate


# ==============================================================================
# RoPE helpers (replicated to avoid version-specific imports)
# ==============================================================================

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """Apply rotary position embedding to a single tensor."""
    return (x * cos) + (_rotate_half(x) * sin)


# ==============================================================================
# CONFIG & OUTPUT WRAPPER
# ==============================================================================

@dataclass
class EdgePruningConfig:
    lambda_edges: float = 1.0
    sparsity_warmup_steps: int = 500
    include_output_edges: bool = True


class _LogitsOutput:
    """Minimal output wrapper with .logits attribute for evaluation compatibility."""
    def __init__(self, logits):
        self.logits = logits


# ==============================================================================
# EDGE-PRUNABLE LLAMA
# ==============================================================================

class EdgePrunableLlama(nn.Module):
    """
    Edge-level pruning for Llama on surviving nodes from node pruning.

    Nodes:
      - Embedding (source only)
      - Attention heads (source + receiver)  — indexed by Q-head
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
        base_model: LlamaForCausalLM,
        active_heads: Dict[int, List[int]],
        active_mlps: Set[int],
        config: EdgePruningConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.edge_config = config
        self.llama_config = base_model.config

        self.active_heads = active_heads
        self.active_mlps = set(active_mlps)

        self.num_layers = self.llama_config.num_hidden_layers
        self.num_heads = self.llama_config.num_attention_heads
        self.num_kv_heads = self.llama_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = self.llama_config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

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
        base = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
        for p in base.parameters():
            p.requires_grad = False
        return cls(base, active_heads, active_mlps, config)

    @property
    def config(self):
        return self.llama_config

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @property
    def dtype(self):
        return next(self.base_model.parameters()).dtype

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
        """Dual-stream edge-gated forward pass for Llama."""
        llama_model = self.base_model.model
        B, S = input_ids.shape
        device = input_ids.device
        dtype = self.dtype
        HD = self.head_dim
        NH = self.num_heads
        NKV = self.num_kv_heads
        H = self.hidden_size
        NKVG = self.num_key_value_groups

        # -- Embeddings (no position emb — RoPE is applied inside attention) --
        clean_emb = llama_model.embed_tokens(input_ids)
        corr_emb = llama_model.embed_tokens(corrupted_input_ids).detach()

        # -- RoPE (computed once, shared across all layers) --
        position_ids = torch.arange(S, device=device).unsqueeze(0)
        position_embeddings = llama_model.rotary_emb(clean_emb, position_ids)
        cos, sin = position_embeddings
        # cos, sin: [1, S, head_dim]
        # For multi-head attention: unsqueeze at head dim
        cos_mh = cos.unsqueeze(1)  # [1, 1, S, HD]
        sin_mh = sin.unsqueeze(1)  # [1, 1, S, HD]

        # -- Causal mask (additive: 0 for attend, -inf for mask) --
        causal = torch.triu(
            torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
        )
        if attention_mask is not None:
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            attend = (~causal).unsqueeze(0) & pad_mask                 # [B, 1, S, S]
        else:
            attend = (~causal).unsqueeze(0).unsqueeze(0)               # [1, 1, S, S]

        float_mask = torch.where(
            attend,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float("-inf"), device=device, dtype=dtype),
        )
        # Per-head mask (squeeze head dim for [B, S, S] broadcasting with single-head scores)
        per_head_mask = float_mask.squeeze(1)  # [B, S, S] or [1, S, S]

        # -- Deltas: delta[i] = clean_source[i] - corrupted_source[i] --
        deltas = [None] * self.num_sources
        deltas[0] = clean_emb - corr_emb

        corrupted_residual = corr_emb

        # -- Layer-by-layer processing --
        for l in range(self.num_layers):
            layer = llama_model.layers[l]
            attn = layer.self_attn
            active_heads_l = self.active_heads.get(l, [])

            # ============================================================
            # CORRUPTED ATTENTION (all heads, needed for corrupted residual)
            # ============================================================
            corr_ln = layer.input_layernorm(corrupted_residual)

            cq = attn.q_proj(corr_ln).view(B, S, NH, HD).transpose(1, 2)    # [B, NH, S, HD]
            ck = attn.k_proj(corr_ln).view(B, S, NKV, HD).transpose(1, 2)   # [B, NKV, S, HD]
            cv = attn.v_proj(corr_ln).view(B, S, NKV, HD).transpose(1, 2)   # [B, NKV, S, HD]

            # Apply RoPE
            cq = _apply_rope(cq, cos_mh, sin_mh)
            ck = _apply_rope(ck, cos_mh, sin_mh)

            # GQA: expand K, V to match Q heads
            ck = ck.repeat_interleave(NKVG, dim=1)  # [B, NH, S, HD]
            cv = cv.repeat_interleave(NKVG, dim=1)  # [B, NH, S, HD]

            c_scores = (cq @ ck.transpose(-2, -1)) / math.sqrt(HD)
            c_scores = c_scores + float_mask
            c_probs = F.softmax(c_scores, dim=-1)
            c_attn_out = c_probs @ cv                                        # [B, NH, S, HD]

            # Full corrupted attention output through o_proj (no bias in Llama)
            c_attn_flat = c_attn_out.transpose(1, 2).reshape(B, S, H)
            c_attn_output = attn.o_proj(c_attn_flat)

            # ============================================================
            # CLEAN ATTENTION (only active heads, edge-gated)
            # ============================================================
            src_before = self.src_idx_before_layer[l]
            if src_before and active_heads_l:
                delta_stack = torch.stack([deltas[s] for s in src_before])

            for h in active_heads_l:
                key = f"L{l}_H{h}"
                kv_h = h // NKVG  # corresponding KV head index

                if src_before:
                    qg = self.attn_q_gates[key]().to(dtype).view(-1, 1, 1, 1)
                    kg = self.attn_k_gates[key]().to(dtype).view(-1, 1, 1, 1)
                    vg = self.attn_v_gates[key]().to(dtype).view(-1, 1, 1, 1)

                    q_in = corrupted_residual + (qg * delta_stack).sum(0)
                    k_in = corrupted_residual + (kg * delta_stack).sum(0)
                    v_in = corrupted_residual + (vg * delta_stack).sum(0)
                else:
                    q_in = k_in = v_in = corrupted_residual

                # Per-head Q / K / V via sliced projection weights
                q_ln = layer.input_layernorm(q_in)
                k_ln = layer.input_layernorm(k_in)
                v_ln = layer.input_layernorm(v_in)

                w_q = attn.q_proj.weight[h * HD : (h + 1) * HD, :]         # [HD, H]
                w_k = attn.k_proj.weight[kv_h * HD : (kv_h + 1) * HD, :]   # [HD, H]
                w_v = attn.v_proj.weight[kv_h * HD : (kv_h + 1) * HD, :]   # [HD, H]

                Q = F.linear(q_ln, w_q)      # [B, S, HD]
                K = F.linear(k_ln, w_k)      # [B, S, HD]
                V = F.linear(v_ln, w_v)      # [B, S, HD]

                # Apply RoPE
                Q = _apply_rope(Q, cos, sin)
                K = _apply_rope(K, cos, sin)

                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(HD)
                scores = scores + per_head_mask
                probs = F.softmax(scores, dim=-1)
                attn_out_h = probs @ V       # [B, S, HD]

                # Output projection slice (no bias)
                w_o = attn.o_proj.weight[:, h * HD : (h + 1) * HD]   # [H, HD]
                clean_h = F.linear(attn_out_h, w_o)                  # [B, S, H]

                # Corrupted per-head output (same slice)
                corr_h = F.linear(c_attn_out[:, h, :, :], w_o)      # [B, S, H]

                src_idx = self.source_to_idx[("head", l, h)]
                deltas[src_idx] = clean_h - corr_h

            # Update corrupted residual with full attention output
            corrupted_residual = corrupted_residual + c_attn_output

            # ============================================================
            # CORRUPTED MLP (full, needed for corrupted residual)
            # ============================================================
            corr_ln2 = layer.post_attention_layernorm(corrupted_residual)
            corr_mlp_out = layer.mlp(corr_ln2)

            # ============================================================
            # CLEAN MLP (if active, edge-gated)
            # ============================================================
            if l in self.active_mlps:
                src_through = self.src_idx_through_attn[l]
                mlp_key = f"L{l}"

                if src_through:
                    mlp_g = self.mlp_edge_gates[mlp_key]().to(dtype).view(-1, 1, 1, 1)
                    ds = torch.stack([deltas[s] for s in src_through])
                    mlp_in = corrupted_residual + (mlp_g * ds).sum(0)
                else:
                    mlp_in = corrupted_residual

                clean_mlp_out = layer.mlp(layer.post_attention_layernorm(mlp_in))

                src_idx = self.source_to_idx[("mlp", l)]
                deltas[src_idx] = clean_mlp_out - corr_mlp_out

            corrupted_residual = corrupted_residual + corr_mlp_out

        # ================================================================
        # OUTPUT
        # ================================================================
        all_deltas = torch.stack(
            [d if d is not None else torch.zeros(B, S, H, device=device, dtype=dtype)
             for d in deltas]
        )

        if self.output_edge_gates is not None:
            og = self.output_edge_gates().to(dtype).view(-1, 1, 1, 1)
            final_residual = corrupted_residual + (og * all_deltas).sum(0)
        else:
            final_residual = corrupted_residual + all_deltas.sum(0)

        logits = self.base_model.lm_head(llama_model.norm(final_residual))
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
