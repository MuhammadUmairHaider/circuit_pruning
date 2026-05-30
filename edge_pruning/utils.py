"""
Utilities for edge-level circuit pruning.

- Extract active nodes from a node-pruned model
- Count dense edges between surviving nodes
- Analyze and report edge pruning results
- Save / load active node specifications
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm


# ==============================================================================
# NODE EXTRACTION
# ==============================================================================

def extract_active_nodes(node_pruned_model) -> Tuple[Dict[int, List[int]], Set[int]]:
    """
    Extract active attention heads and MLP blocks from a node-pruned model
    (after analyze_and_finalize_circuit has been called).

    Returns:
        active_heads: {layer_idx: [head_indices]} for active attention heads
        active_mlps:  set of layer indices with active MLP blocks
    """
    node_pruned_model.eval()

    active_heads = {}
    active_mlps = set()

    # Determine which module list holds the blocks
    if hasattr(node_pruned_model, 'transformer'):
        layers = node_pruned_model.transformer.h
    elif hasattr(node_pruned_model, 'model') and hasattr(node_pruned_model.model, 'layers'):
        layers = node_pruned_model.model.layers
    else:
        raise ValueError("Cannot locate transformer layers in the model")

    with torch.no_grad():
        for l, block in enumerate(layers):
            # -- Attention block --
            attn_active = True
            if hasattr(block, 'attention_block_gate') and block.attention_block_gate is not None:
                attn_active = (block.attention_block_gate() > 0.5).item()

            if attn_active and hasattr(block.attn, 'head_gates') and block.attn.head_gates is not None:
                mask = block.attn.head_gates() > 0.5
                heads = mask.nonzero(as_tuple=True)[0].tolist()
                if heads:
                    active_heads[l] = heads

            # -- MLP block --
            mlp_active = True
            if hasattr(block, 'mlp_block_gate') and block.mlp_block_gate is not None:
                mlp_active = (block.mlp_block_gate() > 0.5).item()
            if mlp_active:
                active_mlps.add(l)

    return active_heads, active_mlps


def save_active_nodes(active_heads, active_mlps, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "active_heads": {str(k): v for k, v in active_heads.items()},
                "active_mlps": sorted(active_mlps),
            },
            f,
            indent=2,
        )
    print(f"Saved active nodes to {path}")


def load_active_nodes(path):
    with open(path) as f:
        data = json.load(f)
    active_heads = {int(k): v for k, v in data["active_heads"].items()}
    active_mlps = set(data["active_mlps"])
    print(f"Loaded active nodes from {path}")
    print(f"  Active heads: {sum(len(v) for v in active_heads.values())} across {len(active_heads)} layers")
    print(f"  Active MLPs: {len(active_mlps)} layers")
    return active_heads, active_mlps


# ==============================================================================
# DENSE EDGE COUNTING (no edge pruning, just surviving nodes)
# ==============================================================================

def count_dense_edges(
    active_heads: Dict[int, List[int]],
    active_mlps: Set[int],
    num_layers: int = 12,
    num_heads_per_layer: int = 12,
    verbose: bool = True,
) -> dict:
    """
    Count edges assuming dense connections between all surviving nodes.
    Mirrors the logic in edgepercent.py.

    Returns dict with total/category edge counts for both full and pruned models.
    """
    total_components_per_layer = num_heads_per_layer + 1  # heads + MLP

    # -- Full model edges --
    full_output_edges = num_layers * total_components_per_layer
    full_mlp_edges, full_qkv_edges = 0, 0
    for j in range(1, num_layers):
        n = j * total_components_per_layer
        full_mlp_edges += n
        full_qkv_edges += num_heads_per_layer * 3 * n

    total_full_original = full_output_edges + full_mlp_edges + full_qkv_edges
    total_full_extra = 1 + num_layers + num_layers * num_heads_per_layer * 6
    total_full = total_full_original + total_full_extra

    # -- Pruned model edges (dense between survivors) --
    # Matches edgepercent.py convention:
    #   "Original" edges: inter-layer connections (sources before layer j)
    #   "Extra" edges: embedding→output, MLP internal, head internal (same-layer)
    head_counts = {l: len(active_heads.get(l, [])) for l in range(num_layers)}

    # Cumulative active sources before each layer
    src_before = {}
    cum = 0
    for i in range(num_layers):
        src_before[i] = cum
        if i in active_mlps:
            cum += 1
        cum += head_counts[i]

    # Count per category
    rem_out_heads = sum(head_counts.values())
    rem_out_mlps = len(active_mlps)
    rem_output = rem_out_heads + rem_out_mlps

    rem_mlp, rem_q, rem_k, rem_v = 0, 0, 0, 0
    for j in range(num_layers):
        n = src_before[j]
        nh = head_counts[j]
        if j in active_mlps:
            rem_mlp += n
        rem_q += nh * n
        rem_k += nh * n
        rem_v += nh * n

    total_rem_original = rem_output + rem_mlp + rem_q + rem_k + rem_v

    rem_extra = 1 + len(active_mlps) + sum(head_counts.values()) * 6
    total_rem = total_rem_original + rem_extra

    result = {
        "full_total": total_full,
        "full_original": total_full_original,
        "full_extra": total_full_extra,
        "dense_total": total_rem,
        "dense_original": total_rem_original,
        "dense_extra": rem_extra,
        "dense_output": rem_output,
        "dense_mlp": rem_mlp,
        "dense_q": rem_q,
        "dense_k": rem_k,
        "dense_v": rem_v,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("  DENSE EDGE COUNT (between surviving nodes)")
        print("=" * 60)
        print(f"Full model edges:  {total_full:,}")
        print(f"Dense edges:       {total_rem:,} / {total_full:,} ({total_rem / total_full:.2%})")
        print(f"  To output:  {rem_output}")
        print(f"  To MLP:     {rem_mlp}")
        print(f"  To Q:       {rem_q}")
        print(f"  To K:       {rem_k}")
        print(f"  To V:       {rem_v}")
        print(f"  Extra:      {rem_extra}")

    return result


# ==============================================================================
# EDGE CIRCUIT ANALYSIS
# ==============================================================================

def analyze_edge_circuit(model, verbose=True) -> dict:
    """
    Analyze edge pruning results. Sets model to final mode, counts active edges
    per category, and prints a detailed report.
    """
    from .models.l0 import HardConcreteGate

    model.eval()
    model.set_final_circuit_mode(True)

    stats = {
        "q_edges": {"total": 0, "active": 0},
        "k_edges": {"total": 0, "active": 0},
        "v_edges": {"total": 0, "active": 0},
        "mlp_edges": {"total": 0, "active": 0},
        "output_edges": {"total": 0, "active": 0},
    }
    per_receiver = {}  # key -> {q_active, k_active, v_active, total}

    with torch.no_grad():
        for key, gate in model.attn_q_gates.items():
            m = gate()
            n, a = m.numel(), int((m > 0.5).sum().item())
            stats["q_edges"]["total"] += n
            stats["q_edges"]["active"] += a
            per_receiver.setdefault(key, {})["q"] = f"{a}/{n}"

        for key, gate in model.attn_k_gates.items():
            m = gate()
            n, a = m.numel(), int((m > 0.5).sum().item())
            stats["k_edges"]["total"] += n
            stats["k_edges"]["active"] += a
            per_receiver.setdefault(key, {})["k"] = f"{a}/{n}"

        for key, gate in model.attn_v_gates.items():
            m = gate()
            n, a = m.numel(), int((m > 0.5).sum().item())
            stats["v_edges"]["total"] += n
            stats["v_edges"]["active"] += a
            per_receiver.setdefault(key, {})["v"] = f"{a}/{n}"

        for key, gate in model.mlp_edge_gates.items():
            m = gate()
            n, a = m.numel(), int((m > 0.5).sum().item())
            stats["mlp_edges"]["total"] += n
            stats["mlp_edges"]["active"] += a
            per_receiver[key] = {"mlp": f"{a}/{n}"}

        if model.output_edge_gates is not None:
            m = model.output_edge_gates()
            n, a = m.numel(), int((m > 0.5).sum().item())
            stats["output_edges"]["total"] = n
            stats["output_edges"]["active"] = a

    total_edges = sum(s["total"] for s in stats.values())
    active_edges = sum(s["active"] for s in stats.values())

    if verbose:
        print("\n" + "=" * 70)
        print("  EDGE PRUNING ANALYSIS")
        print("=" * 70)

        print(f"\nTotal edges:  {total_edges:,}")
        print(f"Active edges: {active_edges:,}")
        if total_edges > 0:
            print(f"Edge reduction: {(total_edges - active_edges) / total_edges * 100:.1f}%")
            print(f"Edge compression: {total_edges / active_edges:.2f}x" if active_edges > 0 else "Edge compression: inf")

        print(f"\nBy category:")
        for cat, s in stats.items():
            name = cat.replace("_", " ").title()
            pct = f"{s['active'] / s['total'] * 100:.1f}%" if s["total"] > 0 else "N/A"
            print(f"  {name}: {s['active']:,} / {s['total']:,} ({pct})")

        print(f"\nPer-receiver detail:")
        for key in sorted(per_receiver.keys()):
            info = per_receiver[key]
            parts = [f"{k.upper()}={v}" for k, v in sorted(info.items())]
            print(f"  {key}: {', '.join(parts)}")

    return {
        "stats": stats,
        "per_receiver": per_receiver,
        "total_edges": total_edges,
        "active_edges": active_edges,
    }


# ==============================================================================
# COMBINED REPORT
# ==============================================================================

def print_combined_report(
    node_stats: dict,
    dense_edge_stats: dict,
    edge_stats: dict,
):
    """Print the three-level compression report."""
    print("\n" + "=" * 70)
    print("  COMBINED COMPRESSION REPORT")
    print("=" * 70)

    # 1. Node compression
    if "granularity_stats" in node_stats:
        gs = node_stats["granularity_stats"]
        ah = gs.get("attention_heads", {})
        mb = gs.get("mlp_blocks", {})
        print(f"\n1. NODE COMPRESSION (from node pruning):")
        if ah.get("total", 0) > 0:
            print(f"   Attention heads: {ah['active']}/{ah['total']} active")
        if mb.get("total", 0) > 0:
            print(f"   MLP blocks:      {mb['active']}/{mb['total']} active")
        if "prunable_compression" in node_stats:
            pc = node_stats["prunable_compression"]
            print(f"   Parameter reduction: {pc['reduction_percentage']:.1f}%")

    # 2. Dense edges
    print(f"\n2. DENSE EDGE COUNT (surviving nodes, no edge pruning):")
    print(f"   Dense edges: {dense_edge_stats['dense_total']:,} / {dense_edge_stats['full_total']:,} "
          f"({dense_edge_stats['dense_total'] / dense_edge_stats['full_total']:.2%} of full model)")

    # 3. Edge compression
    print(f"\n3. EDGE COMPRESSION (after edge pruning):")
    te = edge_stats["total_edges"]
    ae = edge_stats["active_edges"]
    print(f"   Active edges: {ae:,} / {te:,}")
    if te > 0:
        print(f"   Edge reduction: {(te - ae) / te * 100:.1f}%")
    if ae > 0:
        print(f"   Edge compression: {te / ae:.2f}x")
    # Relative to full model
    full = dense_edge_stats["full_total"]
    print(f"   vs full model:  {ae:,} / {full:,} ({ae / full:.2%})")

    print("=" * 70)


# ==============================================================================
# DROPOUT UTILITY
# ==============================================================================

def disable_dropout(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
