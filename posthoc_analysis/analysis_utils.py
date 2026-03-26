"""
Post-hoc analysis utilities for discovered Llama circuits on the IOI task.

Literature basis:
  - Wang et al. (2022)  "Interpretability in the Wild: a Circuit for Indirect Object
    Identification in GPT-2 small"  — head taxonomy, DLA, attention patterns
  - Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"  — OV/QK
    circuits, residual stream decomposition
  - Nostalgebraist (2020) / Belrose et al. (2023) — logit lens

Terminology (IOI):
  IO   — Indirect Object name (the TARGET, the one the model should predict)
  S    — Subject name (the DISTRACTOR, appears twice in the sentence)
  pred_pos — token position just before the final name (where the model predicts)
"""

import os
import sys
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_path():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# 1. Load the circuit model with saved gates applied
# ---------------------------------------------------------------------------

def load_circuit(
    save_dir: str,
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> Tuple:
    """
    Load the PrunableLlamaForCausalLM with saved gate log_alphas injected and
    set to final (hard 0/1 binary mask) mode.  Also loads the matching full model.

    Returns
    -------
    circuit_model, full_model, tokenizer, run_config
    """
    _ensure_path()
    from dataclasses import dataclass
    from transformers import AutoTokenizer, LlamaForCausalLM
    from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
    from models.l0 import HardConcreteGate
    from utils import disable_dropout

    # Inline instead of importing from ioi_llama_kl_budget — that file ends with a
    # bare shell command that causes a SyntaxError on import.
    @dataclass
    class KLBudgetLlamaPruningConfig(PruningConfig):
        init_value: float = 2.0
        sparsity_warmup_steps: int = 1000
        depth_penalty_scaling: float = 0.0
        prune_attention_heads: bool = True
        lambda_attention_heads: float = 1.0
        prune_mlp_hidden: bool = True
        lambda_mlp_hidden: float = 5.0
        prune_mlp_output: bool = True
        lambda_mlp_output: float = 5.0
        prune_attention_neurons: bool = True
        lambda_attention_neurons: float = 1.0
        prune_attention_blocks: bool = True
        lambda_attention_blocks: float = 0.00001
        prune_mlp_blocks: bool = True
        lambda_mlp_blocks: float = 0.00001
        prune_full_layers: bool = True
        lambda_full_layers: float = 0.00001

    # Read run config
    with open(os.path.join(save_dir, "run_config.pkl"), "rb") as f:
        run_config = pickle.load(f)
    model_name = run_config["model"]

    # HF token
    if hf_token is None:
        tok_file = os.path.join(_project_root(), "hf_tokken.txt")
        if os.path.exists(tok_file):
            with open(tok_file) as f:
                hf_token = f.read().strip()

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- circuit model ----
    pruning_config = KLBudgetLlamaPruningConfig()
    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        model_name, pruning_config, **model_kwargs
    ).to(device).eval()
    disable_dropout(circuit_model)

    gate_log_alphas = torch.load(
        os.path.join(save_dir, "gate_log_alphas.pt"), map_location="cpu"
    )
    loaded = 0
    for name, module in circuit_model.named_modules():
        if isinstance(module, HardConcreteGate) and name in gate_log_alphas:
            module.log_alpha.data.copy_(gate_log_alphas[name].float())
            loaded += 1
    print(f"  Loaded {loaded} / {len(gate_log_alphas)} gates")

    for p in circuit_model.parameters():
        p.requires_grad = False
    circuit_model.set_final_circuit_mode(True)
    circuit_model.eval()

    # ---- full reference model ----
    full_model = LlamaForCausalLM.from_pretrained(model_name, **model_kwargs).to(device).eval()
    for p in full_model.parameters():
        p.requires_grad = False

    return circuit_model, full_model, tokenizer, run_config


# ---------------------------------------------------------------------------
# 2. Circuit inventory
# ---------------------------------------------------------------------------

def get_surviving_heads(circuit_model) -> Dict[int, List[int]]:
    """Return {layer_idx: [surviving head indices]}."""
    from models.l0 import HardConcreteGate
    result = {}
    for l, layer in enumerate(circuit_model.model.layers):
        attn = layer.attn
        if attn.head_gates is not None:
            with torch.no_grad():
                gate = attn.head_gates()
            surviving = (gate > 0.5).nonzero(as_tuple=True)[0].tolist()
            result[l] = surviving
        else:
            result[l] = list(range(circuit_model.config.num_attention_heads))
    return result


def get_surviving_mlp_blocks(circuit_model) -> Dict[int, bool]:
    """Return {layer_idx: True/False} for whether the whole MLP block survives."""
    result = {}
    for l, layer in enumerate(circuit_model.model.layers):
        if layer.mlp_block_gate is not None:
            with torch.no_grad():
                g = layer.mlp_block_gate()
            result[l] = bool((g > 0.5).all().item())
        else:
            result[l] = True
    return result


def get_surviving_attn_blocks(circuit_model) -> Dict[int, bool]:
    """Return {layer_idx: True/False} for whether the whole attn block survives."""
    result = {}
    for l, layer in enumerate(circuit_model.model.layers):
        if layer.attention_block_gate is not None:
            with torch.no_grad():
                g = layer.attention_block_gate()
            result[l] = bool((g > 0.5).all().item())
        else:
            result[l] = True
    return result


def get_surviving_mlp_neurons(circuit_model) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Return {layer_idx: (surviving_hidden_indices, surviving_output_indices)}.
    hidden_gates → intermediate neurons (post-SwiGLU)
    output_gates → residual-stream neurons (after down_proj)
    """
    result = {}
    for l, layer in enumerate(circuit_model.model.layers):
        mlp = layer.mlp
        with torch.no_grad():
            hid = (
                (mlp.hidden_gates() > 0.5).cpu().numpy()
                if mlp.hidden_gates is not None
                else np.ones(circuit_model.config.intermediate_size, dtype=bool)
            )
            out = (
                (mlp.output_gates() > 0.5).cpu().numpy()
                if mlp.output_gates is not None
                else np.ones(circuit_model.config.hidden_size, dtype=bool)
            )
        result[l] = (np.where(hid)[0], np.where(out)[0])
    return result


def circuit_summary_table(circuit_model) -> "pd.DataFrame":  # noqa: F821
    """Build a per-layer summary DataFrame."""
    import pandas as pd

    n_layers   = len(circuit_model.model.layers)
    num_heads  = circuit_model.config.num_attention_heads
    inter_size = circuit_model.config.intermediate_size
    hid_size   = circuit_model.config.hidden_size

    surv_heads    = get_surviving_heads(circuit_model)
    surv_attn_blk = get_surviving_attn_blocks(circuit_model)
    surv_mlp_blk  = get_surviving_mlp_blocks(circuit_model)
    surv_neurons  = get_surviving_mlp_neurons(circuit_model)

    rows = []
    for l in range(n_layers):
        heads = surv_heads.get(l, [])
        hid_neurons, out_neurons = surv_neurons.get(l, (np.array([]), np.array([])))
        rows.append({
            "layer":         l,
            "attn_block":    surv_attn_blk.get(l, True),
            "n_heads":       len(heads),
            "head_pct":      len(heads) / num_heads * 100,
            "heads":         heads,
            "mlp_block":     surv_mlp_blk.get(l, True),
            "n_hid_neurons": len(hid_neurons),
            "hid_pct":       len(hid_neurons) / inter_size * 100,
            "n_out_neurons": len(out_neurons),
            "out_pct":       len(out_neurons) / hid_size * 100,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Attention patterns (approximate, via Q·K hooks, no RoPE)
# ---------------------------------------------------------------------------

def find_name_positions_batch(
    input_ids: torch.Tensor,     # (B, S)
    token_ids: torch.Tensor,     # (B,) — one token ID per sample
    search_end: torch.Tensor,    # (B,) — exclusive upper bound (= T_Start, the answer pos)
    bos_offset: int = 1,         # skip BOS at position 0
) -> List[List[int]]:
    """
    For each sample b, find all positions in input_ids[b, bos_offset:search_end[b]]
    where input_ids[b, pos] == token_ids[b].

    Llama tokenizer always adds BOS at position 0, so bos_offset=1 skips it.
    search_end should be T_Start (the 0-indexed position of the answer token)
    so we stay in the sentence body and exclude the final answer.

    Returns list-of-lists: result[b] = sorted list of matching positions.
    """
    B = input_ids.shape[0]
    result = []
    for b in range(B):
        tok  = token_ids[b].item()
        end  = search_end[b].item()
        positions = [
            pos for pos in range(bos_offset, end)
            if input_ids[b, pos].item() == tok
        ]
        result.append(positions)
    return result


def get_attention_patterns(
    circuit_model,
    input_ids:            torch.Tensor,
    attention_mask:       torch.Tensor,
    corrupted_input_ids:  torch.Tensor,   # actual corrupted sentence (names swapped)
    device: str,
    layers: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Compute approximate attention patterns for all (or selected) layers by hooking
    the q_proj and k_proj outputs and computing Q@K.T / sqrt(d_head).

    Uses dual-stream mode with the ACTUAL corrupted inputs so that clean-stream
    residuals reflect what the circuit truly sees at each layer.

    NOTE: RoPE is NOT applied — patterns are a good qualitative approximation of
    which positions each head attends to, but exact magnitudes differ slightly.

    In dual-stream mode q_proj and k_proj each fire TWICE per layer (clean then
    corrupted). We capture only the FIRST call (clean stream).

    Returns
    -------
    Dict[layer_idx -> tensor of shape (batch, num_heads, seq, seq)]
    """
    num_heads    = circuit_model.config.num_attention_heads
    num_kv_heads = circuit_model.config.num_key_value_heads
    head_dim     = circuit_model.config.hidden_size // num_heads
    kv_groups    = num_heads // num_kv_heads
    n_layers     = len(circuit_model.model.layers)
    target_layers = set(layers) if layers is not None else set(range(n_layers))

    q_buf: Dict[int, torch.Tensor] = {}
    k_buf: Dict[int, torch.Tensor] = {}
    hooks = []
    _q_done: Dict[int, bool] = {}
    _k_done: Dict[int, bool] = {}

    for l, layer in enumerate(circuit_model.model.layers):
        if l not in target_layers:
            continue
        _q_done[l] = False
        _k_done[l] = False
        attn = layer.attn.original_attention

        def _make_q(idx):
            def _h(module, inp, out):
                if not _q_done[idx]:        # first call = clean stream
                    q_buf[idx] = out.detach().cpu()
                    _q_done[idx] = True
            return _h

        def _make_k(idx):
            def _h(module, inp, out):
                if not _k_done[idx]:        # first call = clean stream
                    k_buf[idx] = out.detach().cpu()
                    _k_done[idx] = True
            return _h

        hooks.append(attn.q_proj.register_forward_hook(_make_q(l)))
        hooks.append(attn.k_proj.register_forward_hook(_make_k(l)))

    circuit_model.eval()
    with torch.no_grad():
        circuit_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            corrupted_input_ids=corrupted_input_ids.to(device),
        )

    for h in hooks:
        h.remove()

    patterns: Dict[int, torch.Tensor] = {}
    seq_len = input_ids.shape[1]
    am_cpu  = attention_mask.cpu()

    for l in target_layers:
        if l not in q_buf:
            continue
        q = q_buf[l]  # (B, S, num_heads * head_dim)
        k = k_buf[l]  # (B, S, num_kv_heads * head_dim)
        B = q.shape[0]

        q = q.view(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).float()       # B,H,S,D
        k = k.view(B, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3).float()    # B,KV,S,D

        # Expand KV heads for GQA
        k = k.unsqueeze(2).expand(B, num_kv_heads, kv_groups, seq_len, head_dim)
        k = k.reshape(B, num_heads, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # B,H,S,S

        # Causal mask
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding mask
        pad = (am_cpu == 0).unsqueeze(1).unsqueeze(2)   # B,1,1,S
        scores.masked_fill_(pad, float("-inf"))

        patterns[l] = torch.softmax(scores, dim=-1)  # B,H,S,S

    return patterns


# ---------------------------------------------------------------------------
# 4. Direct Logit Attribution (DLA) via residual-stream decomposition
# ---------------------------------------------------------------------------

def _frozen_scale_dla(
    component_output: torch.Tensor,   # (batch, hidden)
    final_residual:   torch.Tensor,   # (batch, hidden) — the full residual at pred pos
    final_ln_weight:  torch.Tensor,   # (hidden,)
    lm_head_weight:   torch.Tensor,   # (vocab, hidden)
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Frozen-scale DLA (TransformerLens style):
        dla = W_U  @  (component_output * (ln_weight / rms(final_residual)))

    This linearises the final RMSNorm around the true residual so each component
    gets a fair share of the logit contribution.

    Returns (batch, vocab)
    """
    rms = final_residual.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    scale = final_ln_weight.float() / rms           # (batch, hidden)
    proj  = component_output.float() * scale        # (batch, hidden)
    return (lm_head_weight.float() @ proj.T).T      # (batch, vocab)


def compute_dla(
    circuit_model,
    input_ids:           torch.Tensor,
    attention_mask:      torch.Tensor,
    corrupted_input_ids: torch.Tensor,   # actual corrupted sentence
    pred_positions:      torch.Tensor,   # (batch,) 0-indexed
    device: str,
    layers: Optional[List[int]] = None,
) -> Dict:
    """
    Compute per-component DLA at the prediction positions.

    Returns dict with keys:
        'attn' : {layer: tensor (batch, num_heads, vocab)}   per-head DLA
        'mlp'  : {layer: tensor (batch, vocab)}              per-layer MLP DLA
        'embed': tensor (batch, vocab)                       embedding DLA
    """
    n_layers  = len(circuit_model.model.layers)
    num_heads = circuit_model.config.num_attention_heads
    head_dim  = circuit_model.config.hidden_size // num_heads
    target_layers = set(layers) if layers is not None else set(range(n_layers))

    mlp_out_buf:        Dict[int, torch.Tensor] = {}
    embed_buf:          Dict[str, torch.Tensor] = {}
    final_residual_buf: Dict[str, torch.Tensor] = {}
    pre_oproj_buf:      Dict[int, torch.Tensor] = {}   # per-head value concat before W_O

    hooks = []
    # In dual-stream mode each of these fires TWICE per layer (clean then corrupted).
    # We want the FIRST call (clean/gated circuit output).
    _embed_done = [False]
    _oproj_done: Dict[int, bool] = {}
    _mlp_done:   Dict[int, bool] = {}

    # --- embedding hook — first call = clean embeddings ---
    def _embed_hook(module, inp, out):
        if not _embed_done[0]:
            embed_buf["embed"] = out.detach().cpu()
            _embed_done[0] = True
    hooks.append(circuit_model.model.embed_tokens.register_forward_hook(_embed_hook))

    # --- final residual (input to final norm) — called once on clean stream only ---
    def _final_norm_hook(module, inp, out):
        final_residual_buf["residual"] = inp[0].detach().cpu()
    hooks.append(circuit_model.model.norm.register_forward_hook(_final_norm_hook))

    # --- per-layer hooks ---
    for l, layer in enumerate(circuit_model.model.layers):
        if l not in target_layers:
            continue
        _oproj_done[l] = False
        _mlp_done[l]   = False

        # Hook o_proj INPUT to get per-head value outputs (V @ W_O decomposition).
        # First call = gated clean circuit output; second = corrupted stream.
        def _make_pre_oproj(idx):
            def _h(module, inp, out):
                if not _oproj_done[idx]:
                    pre_oproj_buf[idx] = inp[0].detach().cpu()   # (B, S, H*D)
                    _oproj_done[idx] = True
            return _h
        hooks.append(layer.attn.original_attention.o_proj.register_forward_hook(_make_pre_oproj(l)))

        # MLP output (after down_proj).  First call = gated circuit output.
        def _make_mlp_out(idx):
            def _h(module, inp, out):
                if not _mlp_done[idx]:
                    mlp_out_buf[idx] = out.detach().cpu()   # (B, S, hidden)
                    _mlp_done[idx] = True
            return _h
        hooks.append(layer.mlp.original_mlp.down_proj.register_forward_hook(_make_mlp_out(l)))

    circuit_model.eval()
    with torch.no_grad():
        circuit_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            corrupted_input_ids=corrupted_input_ids.to(device),
        )

    for h in hooks:
        h.remove()

    # --- compute DLA ---
    final_residual = final_residual_buf["residual"]          # (B, S, hidden)
    ln_w = circuit_model.model.norm.weight.detach().cpu()    # (hidden,)
    lm_w = circuit_model.lm_head.weight.detach().cpu()       # (vocab, hidden)
    W_O  = circuit_model.model.layers[0].attn.original_attention.o_proj.weight.detach().cpu()  # placeholder

    batch_size = input_ids.shape[0]
    pred_pos   = pred_positions.cpu()  # (B,)

    def _at_pred(tensor):
        """Index (B,S,...) -> (B,...) at the prediction position per sample."""
        return torch.stack([tensor[b, pred_pos[b]] for b in range(batch_size)])

    final_res_pred = _at_pred(final_residual)   # (B, hidden)

    result = {"attn": {}, "mlp": {}, "embed": None}

    # Embedding DLA
    if "embed" in embed_buf:
        emb_pred = _at_pred(embed_buf["embed"])
        result["embed"] = _frozen_scale_dla(emb_pred, final_res_pred, ln_w, lm_w)

    for l in sorted(target_layers):
        # --- per-head attention DLA ---
        if l in pre_oproj_buf:
            pre_op = _at_pred(pre_oproj_buf[l])  # (B, H*D)
            W_O_l  = circuit_model.model.layers[l].attn.original_attention.o_proj.weight.detach().cpu()
            # (hidden, H*D)
            head_dlas = []
            for h in range(num_heads):
                v_h   = pre_op[:, h * head_dim : (h + 1) * head_dim]   # (B, D)
                W_O_h = W_O_l[:, h * head_dim : (h + 1) * head_dim]    # (hidden, D)
                contrib = (W_O_h @ v_h.T).T                             # (B, hidden)
                head_dlas.append(
                    _frozen_scale_dla(contrib, final_res_pred, ln_w, lm_w)
                )                                                        # (B, vocab)
            result["attn"][l] = torch.stack(head_dlas, dim=1)           # (B, H, vocab)

        # --- MLP DLA ---
        if l in mlp_out_buf:
            mlp_pred = _at_pred(mlp_out_buf[l])                         # (B, hidden)
            result["mlp"][l] = _frozen_scale_dla(mlp_pred, final_res_pred, ln_w, lm_w)

    return result


def dla_io_vs_s(
    dla_tensor: torch.Tensor,       # (batch, vocab) or (batch, heads, vocab)
    target_ids: torch.Tensor,       # (batch,)
    distractor_ids: torch.Tensor,   # (batch,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the IO (target) and S (distractor) DLA scores for each sample.
    Returns (io_score, s_score) both shape (batch,) or (batch, heads).
    """
    B = dla_tensor.shape[0]
    if dla_tensor.dim() == 2:  # (B, vocab)
        io  = dla_tensor[torch.arange(B), target_ids]
        s   = dla_tensor[torch.arange(B), distractor_ids]
    else:                       # (B, H, vocab)
        io  = dla_tensor[torch.arange(B), :, target_ids].T   # (H, B) → wait
        # Correct indexing: dla_tensor[b, h, target_ids[b]]
        io = torch.stack([dla_tensor[b, :, target_ids[b]] for b in range(B)])   # (B, H)
        s  = torch.stack([dla_tensor[b, :, distractor_ids[b]] for b in range(B)])
    return io, s


# ---------------------------------------------------------------------------
# 5. OV circuit: what does each head promote in logit space?
# ---------------------------------------------------------------------------

def compute_ov_top_tokens(
    circuit_model,
    layer: int,
    head: int,
    tokenizer,
    top_k: int = 15,
) -> Tuple[List[str], List[str], float]:
    """
    Compute W_U @ W_OV_h @ W_E to find which tokens each head promotes/suppresses.

    For each input token t, the head (when attending to t with weight 1) contributes:
        contribution = W_O_h @ W_V_h @ embed(t)   [shape: hidden]
        logit_boost  = W_U @ contribution           [shape: vocab]

    We report:
        - top-k tokens where the DIAGONAL of W_U @ W_OV_h @ W_E is largest
          (= tokens the head copies from input to output)
        - copying_score: mean diagonal / mean off-diagonal of that matrix

    Returns (top_promoted_tokens, top_suppressed_tokens, copying_score)
    """
    attn  = circuit_model.model.layers[layer].attn.original_attention
    head_dim     = circuit_model.config.hidden_size // circuit_model.config.num_attention_heads
    num_kv_heads = circuit_model.config.num_key_value_heads
    kv_groups    = circuit_model.config.num_attention_heads // num_kv_heads
    kv_head      = head // kv_groups

    W_V_h = attn.v_proj.weight[kv_head * head_dim : (kv_head + 1) * head_dim, :].detach().float().cpu()
    # (head_dim, hidden)
    W_O_h = attn.o_proj.weight[:, head * head_dim : (head + 1) * head_dim].detach().float().cpu()
    # (hidden, head_dim)

    W_E = circuit_model.model.embed_tokens.weight.detach().float().cpu()   # (vocab, hidden)
    W_U = circuit_model.lm_head.weight.detach().float().cpu()              # (vocab, hidden)

    W_OV  = W_O_h @ W_V_h          # (hidden, hidden)
    # Full mapping: input token t → output logits
    # M[out, in] = W_U[out, :] @ W_OV @ W_E[in, :]
    # Too large (vocab × vocab) — compute diagonal only for copying score
    # and top overall rows/cols for interpretation

    # Copying score (diagonal of W_U @ W_OV @ W_E)
    # = (W_U @ W_OV) element-wise dot (W_E) summed over hidden — i.e. diag of M
    W_U_OV = W_U @ W_OV            # (vocab, hidden)
    diag   = (W_U_OV * W_E).sum(-1)  # (vocab,) — M[t, t] for each token t

    top_copy_idx = diag.topk(top_k).indices.tolist()
    top_copy_tokens = [tokenizer.decode([i]).strip() for i in top_copy_idx]

    # Tokens that the head promotes on AVERAGE across all inputs
    # = W_U @ W_OV @ mean(W_E) — but more useful: find the direction W_OV points to
    mean_W_E = W_E.mean(0)          # (hidden,)
    avg_out  = W_OV @ mean_W_E      # (hidden,)
    avg_logits = W_U @ avg_out      # (vocab,)
    top_prom_idx = avg_logits.topk(top_k).indices.tolist()
    top_supp_idx = avg_logits.topk(top_k, largest=False).indices.tolist()

    top_promoted  = [tokenizer.decode([i]).strip() for i in top_prom_idx]
    top_suppressed = [tokenizer.decode([i]).strip() for i in top_supp_idx]

    # Copying score: mean diagonal vs mean absolute value
    copying_score = diag.mean().item() / (W_U_OV.abs().mean().item() + 1e-9)

    return top_copy_tokens, top_promoted, top_suppressed, copying_score


# ---------------------------------------------------------------------------
# 6. Ablation: zero a specific component and measure accuracy impact
# ---------------------------------------------------------------------------

def _set_head_gate(circuit_model, layer: int, head: int, value: float):
    """Directly set the log_alpha for a head gate so it evaluates to `value`."""
    gate = circuit_model.model.layers[layer].attn.head_gates
    if gate is None:
        return
    # Force the gate to exactly `value` by setting log_alpha accordingly:
    # gate = sigmoid(log_alpha) * (zeta - gamma) + gamma, hard-clamped to [0,1]
    # For value=0: set log_alpha << 0  → -100
    # For value=1: set log_alpha >> 0  → +100
    gate.log_alpha.data[head] = -100.0 if value == 0.0 else 100.0


def ablate_head_and_eval(
    circuit_model,
    layer: int,
    head: int,
    dataloader,
    device: str,
) -> float:
    """
    Temporarily zero the gate for (layer, head) and return accuracy on the dataloader.
    Restores original value after evaluation.
    """
    from dataset.ioi_llama import run_evaluation

    gate = circuit_model.model.layers[layer].attn.head_gates
    if gate is None:
        return None

    orig_val = gate.log_alpha.data[head].item()
    gate.log_alpha.data[head] = -100.0  # force to 0

    circuit_model.eval()
    results = run_evaluation(
        model_to_eval=circuit_model,
        model_name=f"ablate_L{layer}H{head}",
        full_model_for_faithfulness=None,
        dataloader=dataloader,
        device=device,
        verbose=False,
        tokenizer=None,
    )

    gate.log_alpha.data[head] = orig_val  # restore
    return results["accuracy"]


def ablate_mlp_block_and_eval(circuit_model, layer: int, dataloader, device: str) -> float:
    """Zero the mlp_block_gate for a layer and return accuracy."""
    from dataset.ioi_llama import run_evaluation

    bg = circuit_model.model.layers[layer].mlp_block_gate
    if bg is None:
        return None
    orig = bg.log_alpha.data.clone()
    bg.log_alpha.data.fill_(-100.0)

    circuit_model.eval()
    results = run_evaluation(
        model_to_eval=circuit_model,
        model_name=f"ablate_mlp_L{layer}",
        full_model_for_faithfulness=None,
        dataloader=dataloader,
        device=device,
        verbose=False,
        tokenizer=None,
    )

    bg.log_alpha.data.copy_(orig)
    return results["accuracy"]


# ---------------------------------------------------------------------------
# 7. Logit lens: project residual stream at each layer to vocabulary
# ---------------------------------------------------------------------------

def compute_logit_lens(
    circuit_model,
    input_ids:           torch.Tensor,
    attention_mask:      torch.Tensor,
    corrupted_input_ids: torch.Tensor,   # actual corrupted sentence
    pred_positions:      torch.Tensor,   # (batch,) 0-indexed
    device: str,
    topk: int = 5,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    At each layer, apply the final LN + unembedding to the residual stream at the
    prediction positions, returning the top-k predicted tokens.

    Uses dual-stream mode with actual corrupted inputs so layer outputs reflect
    the true circuit residuals.  Layer hook out[0] = final_clean per-layer state.
    embed_tokens fires twice (clean then corrupted); first call is captured.

    Returns Dict[layer_idx -> dict] where layer -1 = after embedding.
    """
    residuals: Dict[int, torch.Tensor] = {}
    hooks = []
    _embed_done = [False]

    # Layer -1: after embedding — fires twice, capture first (clean)
    def _embed_hook(module, inp, out):
        if not _embed_done[0]:
            residuals[-1] = out.detach().cpu()
            _embed_done[0] = True
    hooks.append(circuit_model.model.embed_tokens.register_forward_hook(_embed_hook))

    # After each decoder block — fires once, out = (final_clean, final_corrupted, attn)
    for l, layer in enumerate(circuit_model.model.layers):
        def _make_layer_hook(idx):
            def _h(module, inp, out):
                # Dual-stream returns (final_clean, final_corrupted, attn_outputs)
                hs = out[0] if isinstance(out, tuple) else out
                residuals[idx] = hs.detach().cpu()
            return _h
        hooks.append(layer.register_forward_hook(_make_layer_hook(l)))

    circuit_model.eval()
    with torch.no_grad():
        circuit_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            corrupted_input_ids=corrupted_input_ids.to(device),
        )

    for h in hooks:
        h.remove()

    final_ln = circuit_model.model.norm
    lm_head  = circuit_model.lm_head
    batch_size = input_ids.shape[0]
    pred_pos   = pred_positions.cpu()

    def _at_pred(tensor):
        return torch.stack([tensor[b, pred_pos[b]] for b in range(batch_size)])

    result = {}
    for l_idx in sorted(residuals.keys()):
        res_pred = _at_pred(residuals[l_idx]).to(device)
        with torch.no_grad():
            normed   = final_ln(res_pred)
            logits   = lm_head(normed)              # (B, vocab)
            probs    = logits.softmax(-1)
            top_vals, top_ids = probs.topk(topk, dim=-1)  # (B, topk)
        # Average across batch
        avg_logits = logits.mean(0)                 # (vocab,)
        top_l_vals, top_l_ids = avg_logits.topk(topk)
        result[l_idx] = {
            "top_ids":  top_l_ids.cpu().tolist(),
            "top_logits": top_l_vals.cpu().tolist(),
            "per_sample_top_ids": top_ids.cpu().tolist(),
        }

    return result


# ---------------------------------------------------------------------------
# 8. IOI-specific attention pattern scores
# ---------------------------------------------------------------------------

def ioi_attention_scores(
    attn_patterns: Dict[int, torch.Tensor],   # {layer: (B, H, S, S)}
    pred_positions: torch.Tensor,             # (B,) — where we measure FROM
    io_positions:   torch.Tensor,             # (B,) — IO name token position
    s_positions:    List[torch.Tensor],       # list of length 2: [S1_pos (B,), S2_pos (B,)]
    seq_len: int,
) -> Dict[int, torch.Tensor]:
    """
    For each layer, compute per-head "IO attention score" and "S attention score"
    at the prediction position.

    Returns Dict[layer -> tensor (B, H, 4)] where dim-2 is:
        [io_attn, s1_attn, s2_attn, other_attn]
    """
    B = pred_positions.shape[0]
    result = {}

    for l, patterns in attn_patterns.items():
        # patterns: (B, H, S, S)
        # We want the row at pred_position
        H = patterns.shape[1]
        scores = torch.zeros(B, H, 4)

        for b in range(B):
            row = patterns[b, :, pred_positions[b], :]  # (H, S)
            # IO attention
            io_p = io_positions[b].item()
            if 0 <= io_p < seq_len:
                scores[b, :, 0] = row[:, io_p]
            # S1 attention
            s1_p = s_positions[0][b].item()
            if 0 <= s1_p < seq_len:
                scores[b, :, 1] = row[:, s1_p]
            # S2 attention
            s2_p = s_positions[1][b].item()
            if 0 <= s2_p < seq_len:
                scores[b, :, 2] = row[:, s2_p]
            # Other (1 - sum of above)
            scores[b, :, 3] = 1.0 - scores[b, :, 0] - scores[b, :, 1] - scores[b, :, 2]

        result[l] = scores  # (B, H, 4)

    return result


def classify_head_role(
    io_attn: float,
    s_attn: float,
    io_dla: float,
    s_dla: float,
) -> str:
    """
    Classify a head's role based on its mean attention pattern and DLA scores.

    Head taxonomy (adapted from Wang et al. 2022):
        IO-Mover (NMH)    : high IO attention, positive IO DLA
        S-Inhibitor       : high S attention, negative S DLA (suppresses subject)
        Negative NMH      : high IO attention, negative IO DLA (suppresses IO)
        Duplicate-Token   : high S1+S2 attention (sees repeated token)
        Other             : doesn't fit cleanly
    """
    io_dominant = io_attn > s_attn
    s_dominant  = s_attn  > io_attn

    if io_dominant and io_dla > 0:
        return "IO-Mover"
    if io_dominant and io_dla < 0:
        return "Neg-IO-Mover"
    if s_dominant and s_dla < 0:
        return "S-Inhibitor"
    if s_dominant and s_dla > 0:
        return "S-Promoter"
    return "Other"


# ===========================================================================
# 9. Neuron-level analysis
# ===========================================================================

def get_mlp_neuron_vocab_projections(
    circuit_model,
    layer: int,
    neuron_indices: List[int],
    tokenizer,
    top_k: int = 15,
) -> Dict[int, Dict]:
    """
    Weight-space analysis for surviving MLP hidden neurons in a given layer.

    For each neuron i (Llama SwiGLU):
      - WRITE direction  : down_proj[:, i]           (what it adds to residual)
      - Vocab projection : W_U @ down_proj[:, i]     (which tokens it promotes/suppresses)
      - Gate sensitivity : gate_proj[i, :] @ W_E.T   (which input tokens activate the gate)
      - Up   sensitivity : up_proj[i, :] @ W_E.T     (which input tokens modulate magnitude)

    Returns Dict[neuron_idx -> dict of analysis results]
    """
    mlp  = circuit_model.model.layers[layer].mlp.original_mlp
    W_U  = circuit_model.lm_head.weight.detach().float().cpu()           # (V, H)
    W_E  = circuit_model.model.embed_tokens.weight.detach().float().cpu()# (V, H)
    W_dn = mlp.down_proj.weight.detach().float().cpu()                   # (H, I)
    W_g  = mlp.gate_proj.weight.detach().float().cpu()                   # (I, H)
    W_up = mlp.up_proj.weight.detach().float().cpu()                     # (I, H)

    result = {}
    for i in neuron_indices:
        write_dir    = W_dn[:, i]                      # (H,)
        write_logits = W_U @ write_dir                 # (V,)

        gate_sens = W_g[i] @ W_E.T                    # (V,) — which tokens activate gate
        up_sens   = W_up[i] @ W_E.T                   # (V,) — which tokens scale up path

        result[i] = {
            "write_logits":        write_logits,
            "top_promoted":        [tokenizer.decode([j]).strip() for j in write_logits.topk(top_k).indices.tolist()],
            "top_suppressed":      [tokenizer.decode([j]).strip() for j in write_logits.topk(top_k, largest=False).indices.tolist()],
            "top_gate_activating": [tokenizer.decode([j]).strip() for j in gate_sens.topk(top_k).indices.tolist()],
            "top_up_activating":   [tokenizer.decode([j]).strip() for j in up_sens.topk(top_k).indices.tolist()],
            "top_gate_suppressing":[tokenizer.decode([j]).strip() for j in gate_sens.topk(top_k, largest=False).indices.tolist()],
            "write_norm":          write_dir.norm().item(),
        }
    return result


def capture_mlp_neuron_activations(
    circuit_model,
    input_ids:           torch.Tensor,
    attention_mask:      torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    layer: int,
    pred_positions:      torch.Tensor,   # (B,) 0-indexed
    device: str,
) -> torch.Tensor:
    """
    Capture SwiGLU intermediate activations for each MLP neuron at the prediction
    positions.  Hooks the input of down_proj (first call in dual-stream = clean
    stream after hidden_gate masking).

    Returns tensor of shape (batch, intermediate_size).
    Pruned neurons will have activation 0 (zeroed by hidden_gates).
    """
    act_cache  = {}
    call_count = [0]

    def _hook(module, inp, out):
        if call_count[0] == 0:
            act_cache["act"] = inp[0].detach().cpu()   # (B, S, I)
        call_count[0] += 1

    mlp = circuit_model.model.layers[layer].mlp.original_mlp
    h   = mlp.down_proj.register_forward_hook(_hook)

    circuit_model.eval()
    with torch.no_grad():
        circuit_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            corrupted_input_ids=corrupted_input_ids.to(device),
        )
    h.remove()

    if "act" not in act_cache:
        return None

    act = act_cache["act"]        # (B, S, I)
    B   = act.shape[0]
    return torch.stack([act[b, pred_positions[b], :] for b in range(B)])  # (B, I)


def get_attn_neuron_vocab_projections(
    circuit_model,
    layer: int,
    head: int,
    surviving_dims: List[int],   # indices within [0, head_dim)
    tokenizer,
    top_k: int = 15,
) -> Dict[int, Dict]:
    """
    For each surviving attention-neuron dimension d in head h:
      - Write direction  : o_proj[:, h*head_dim + d]
      - Vocab projection : W_U @ write_dir

    The neuron_gates operate across ALL dimensions of ALL heads stacked
    (shape num_heads * head_dim), so dim d in head h is global index h*head_dim+d.
    surviving_dims should be local indices within [0, head_dim).
    """
    head_dim = circuit_model.config.hidden_size // circuit_model.config.num_attention_heads
    attn = circuit_model.model.layers[layer].attn.original_attention
    W_O  = attn.o_proj.weight.detach().float().cpu()    # (H_size, num_heads * head_dim)
    W_U  = circuit_model.lm_head.weight.detach().float().cpu()

    result = {}
    for d in surviving_dims:
        global_d  = head * head_dim + d
        write_dir = W_O[:, global_d]                    # (H_size,)
        write_logits = W_U @ write_dir

        result[d] = {
            "top_promoted":   [tokenizer.decode([j]).strip() for j in write_logits.topk(top_k).indices.tolist()],
            "top_suppressed": [tokenizer.decode([j]).strip() for j in write_logits.topk(top_k, largest=False).indices.tolist()],
            "write_norm":     write_dir.norm().item(),
            "write_logits":   write_logits,
        }
    return result


def compute_neuron_io_correlation(
    activations:   torch.Tensor,   # (N_examples, intermediate_size)
    logit_diffs:   torch.Tensor,   # (N_examples,) — IO logit − S logit
    neuron_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Pearson correlation between each neuron's activation and the IO−S logit
    difference across examples.

    Returns (intermediate_size,) correlation vector.
    Positive = neuron tends to be active when model is confident in IO answer.
    """
    if neuron_indices is not None:
        acts = activations[:, neuron_indices].float()
    else:
        acts = activations.float()

    ld = logit_diffs.float()

    # Standardise
    acts_z = acts - acts.mean(0, keepdim=True)
    ld_z   = ld   - ld.mean()
    denom_acts = acts_z.norm(dim=0).clamp(min=1e-8)
    denom_ld   = ld_z.norm().clamp(min=1e-8)

    corr = (ld_z.unsqueeze(1) * acts_z).sum(0) / (denom_acts * denom_ld)
    return corr   # (n_neurons,)


def capture_logit_diffs_batch(
    circuit_model,
    dataloader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the circuit model over the dataloader and collect per-example
    IO logit − S logit at the prediction position.

    Returns (logit_diffs, correct_mask) both shape (N,).
    """
    lds, correct = [], []

    circuit_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            out = circuit_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                corrupted_input_ids=batch["corrupted_input_ids"].to(device),
            )
            # In dual-stream mode, out is a CausalLMOutputWithPast-like object
            # but the circuit forward returns a custom tuple; handle both
            if hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out[0]

            B = logits.shape[0]
            for b in range(B):
                t_start = batch["T_Start"][b].item() - 1   # pred pos (0-indexed)
                tgt     = batch["target_tokens"][b, 0].item()
                dist    = batch["distractor_tokens"][b, 0].item()
                lv      = logits[b, t_start]
                ld      = (lv[tgt] - lv[dist]).cpu().item()
                lds.append(ld)
                correct.append(int(lv.argmax().item() == tgt))

    return torch.tensor(lds), torch.tensor(correct)
