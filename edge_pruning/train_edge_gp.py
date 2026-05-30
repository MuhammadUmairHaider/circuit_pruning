"""
Two-phase circuit discovery for GPT-2 on the Gender Pronouns (GP) task:
  Phase 1: Node pruning (multi-granularity) — discovers which heads / MLPs matter
  Phase 2: Edge pruning — discovers which connections between surviving nodes matter

Usage (from the circuit_pruning/ directory):
  python -m edge_pruning.train_edge_gp                     # run both phases
  python -m edge_pruning.train_edge_gp --skip-node-pruning # load saved nodes, edge pruning only
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- Ensure parent directory is on path --
_PARENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from edge_pruning.dataset.gp import (
    GPDataset,
    load_or_generate_gp_data,
    run_evaluation,
    filter_dataset_by_model_correctness,
)
from models.gpt2_circuit import PrunableGPT2LMHeadModel as NodePrunableGPT2
from models.gpt2_circuit import PruningConfig  # node-level config from models/
from utils import analyze_and_finalize_circuit as analyze_node_circuit

from edge_pruning.models.gpt2_edge_circuit import EdgePrunableGPT2, EdgePruningConfig
from edge_pruning.utils import (
    extract_active_nodes,
    save_active_nodes,
    load_active_nodes,
    count_dense_edges,
    analyze_edge_circuit,
    print_combined_report,
    disable_dropout,
)


# ==============================================================================
# GPU MEMORY TRACKER
# ==============================================================================

class GPUMemoryTracker:
    """
    Records GPU memory snapshots at labelled checkpoints and prints
    a detailed comparison table at the end.
    """

    def __init__(self):
        self.snapshots = []
        self._enabled = torch.cuda.is_available()
        if self._enabled:
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _gb(nbytes):
        return nbytes / 1024 ** 3

    def snap(self, tag: str):
        if not self._enabled:
            return
        a = torch.cuda.memory_allocated()
        r = torch.cuda.memory_reserved()
        p = torch.cuda.max_memory_allocated()
        self.snapshots.append((tag, a, r, p))
        print(f"  [GPU] {tag}: "
              f"alloc {self._gb(a):.2f} GB | "
              f"reserved {self._gb(r):.2f} GB | "
              f"peak {self._gb(p):.2f} GB")

    def reset_peak(self):
        if self._enabled:
            torch.cuda.reset_peak_memory_stats()

    def print_report(self):
        if not self._enabled or not self.snapshots:
            return

        print("\n" + "=" * 90)
        print("  GPU MEMORY MAP")
        print("=" * 90)

        print(f"{'Step':<40} {'Alloc (GB)':>10} {'Delta':>10} "
              f"{'Reserved':>10} {'Peak':>10}")
        print("-" * 90)

        prev_alloc = 0
        for tag, a, r, p in self.snapshots:
            delta = a - prev_alloc
            sign = "+" if delta >= 0 else ""
            print(f"{tag:<40} {self._gb(a):>10.3f} {sign + f'{self._gb(delta):.3f}':>10} "
                  f"{self._gb(r):>10.3f} {self._gb(p):>10.3f}")
            prev_alloc = a

        print("-" * 90)

        node_peaks = [p for t, a, r, p in self.snapshots if "node" in t.lower()]
        edge_peaks = [p for t, a, r, p in self.snapshots if "edge" in t.lower()]

        total_gpu = torch.cuda.get_device_properties(0).total_memory

        print(f"\n{'Component':<40} {'Peak Alloc (GB)':>15} {'% of GPU':>10}")
        print("-" * 65)

        full_snaps = [(t, a) for t, a, r, p in self.snapshots if "full model" in t.lower()]
        if full_snaps:
            full_alloc = full_snaps[0][1]
            print(f"{'Full model (frozen, always resident)':<40} "
                  f"{self._gb(full_alloc):>15.3f} {full_alloc/total_gpu*100:>9.1f}%")

        if node_peaks:
            np_ = max(node_peaks)
            print(f"{'Node pruning (peak during training)':<40} "
                  f"{self._gb(np_):>15.3f} {np_/total_gpu*100:>9.1f}%")

        if edge_peaks:
            ep_ = max(edge_peaks)
            print(f"{'Edge pruning (peak during training)':<40} "
                  f"{self._gb(ep_):>15.3f} {ep_/total_gpu*100:>9.1f}%")

        overall_peak = max(p for _, _, _, p in self.snapshots)
        print(f"{'Overall peak':<40} "
              f"{self._gb(overall_peak):>15.3f} {overall_peak/total_gpu*100:>9.1f}%")
        print(f"{'GPU total':<40} {self._gb(total_gpu):>15.3f}")

        if node_peaks and edge_peaks:
            diff = max(edge_peaks) - max(node_peaks)
            sign = "+" if diff >= 0 else ""
            print(f"\nEdge pruning extra over node pruning peak: "
                  f"{sign}{self._gb(diff):.3f} GB")

        print("=" * 90)


_tracker = GPUMemoryTracker()


# ==============================================================================
# NODE PRUNING CONFIG
# ==============================================================================

PRUNING_FACTOR = 1.0


@dataclass
class LocalNodePruningConfig:
    """Local copy so you can tune node pruning independently of models/gpt2_circuit.py."""
    init_value: float = 0.5
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.0
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0 * PRUNING_FACTOR
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0 * PRUNING_FACTOR
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0 * PRUNING_FACTOR
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0 * PRUNING_FACTOR
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 2.0 * PRUNING_FACTOR
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0 * PRUNING_FACTOR
    prune_embedding: bool = False
    lambda_embedding: float = 1.0 * PRUNING_FACTOR


# ==============================================================================
# PHASE 1: NODE PRUNING
# ==============================================================================

def run_node_pruning(
    model_name, tokenizer, full_model,
    train_dataloader, val_dataloader, test_dataloader,
    device, num_epochs=500, lr=3e-1, lambda_sparsity=0.99,
):
    """Run node-level circuit pruning for the GP task."""
    print("\n" + "=" * 70)
    print("  PHASE 1: NODE PRUNING (Gender Pronouns)")
    print("=" * 70)

    pruning_config = LocalNodePruningConfig()
    circuit_model = NodePrunableGPT2.from_pretrained_with_pruning(
        model_name, pruning_config
    ).to(device).eval()
    disable_dropout(circuit_model)

    # Freeze base, unfreeze gates
    for name, param in circuit_model.named_parameters():
        param.requires_grad = "gate" in name

    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    _tracker.reset_peak()
    _tracker.snap("Node model loaded")

    circuit_model.train()
    total_steps = 0
    start = time.time()
    pbar = tqdm(range(num_epochs), desc="Node pruning")

    for epoch in pbar:
        ep_loss, ep_kl, ep_sp = 0.0, 0.0, 0.0

        for batch in train_dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            circuit_out = circuit_model(
                input_ids=batch["input_ids"],
                corrupted_input_ids=batch["corrupted_input_ids"],
                attention_mask=batch["attention_mask"],
            )
            with torch.no_grad():
                target_out = full_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            # KL loss at prediction position (prefix_length - 1)
            bs = circuit_out.logits.size(0)
            total_kl = 0
            for i in range(bs):
                pred_pos = batch["prefix_length"][i] - 1

                circuit_logits = circuit_out.logits[i, pred_pos, :]
                target_logits = target_out.logits[i, pred_pos, :]

                kl = F.kl_div(
                    F.log_softmax(circuit_logits, dim=-1),
                    F.log_softmax(target_logits, dim=-1),
                    reduction="sum", log_target=True,
                )
                total_kl += kl
            kl_loss = total_kl / bs

            # Task loss: target pronoun logit should exceed distractor
            logit_good = circuit_out.logits[
                torch.arange(bs, device=device),
                batch["prefix_length"] - 1,
                batch["target_token"]
            ]
            logit_bad = circuit_out.logits[
                torch.arange(bs, device=device),
                batch["prefix_length"] - 1,
                batch["distractor_token"]
            ]
            task_loss = F.relu(0.1 - (logit_good - logit_bad)).mean()

            sp_loss = circuit_model.get_sparsity_loss(step=total_steps)["total_sparsity"]

            loss = (1 - lambda_sparsity) * (kl_loss + task_loss) + lambda_sparsity * sp_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_kl += kl_loss.item()
            ep_sp += sp_loss.item()
            total_steps += 1

        scheduler.step()
        n = len(train_dataloader)
        pbar.set_postfix(L=f"{ep_loss/n:.3f}", KL=f"{ep_kl/n:.3f}",
                         Sp=f"{ep_sp/n:.3f}")

        if epoch == 0:
            _tracker.snap("Node epoch 1 (fwd+bwd+optim)")

        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            run_evaluation(circuit_model, f"Node Ep {epoch+1}", full_model,
                           val_dataloader, device, tokenizer=tokenizer)
            circuit_model.train()

    print(f"Node pruning time: {time.time() - start:.1f}s")
    _tracker.snap("Node pruning done")

    # Enable full layer pruning for final analysis (matches gp.py)
    pruning_config.prune_full_layers = True
    circuit_model.set_pruning_config(pruning_config)

    # Finalize
    node_stats = analyze_node_circuit(circuit_model)

    return circuit_model, node_stats


# ==============================================================================
# PHASE 2: EDGE PRUNING
# ==============================================================================

def run_edge_pruning(
    model_name, tokenizer, full_model,
    active_heads, active_mlps,
    train_dataloader, val_dataloader, test_dataloader,
    device, num_epochs=300, lr=3e-2, lambda_sparsity=0.90,
):
    """Run edge-level circuit pruning on surviving nodes for the GP task."""
    print("\n" + "=" * 70)
    print("  PHASE 2: EDGE PRUNING (Gender Pronouns)")
    print("=" * 70)

    edge_config = EdgePruningConfig(
        lambda_edges=1.0,
        sparsity_warmup_steps=500,
        include_output_edges=True,
    )

    edge_model = EdgePrunableGPT2.from_pretrained_with_edges(
        model_name, active_heads, active_mlps, edge_config
    ).to(device)
    disable_dropout(edge_model)

    # Only edge gates are trainable
    trainable = sum(p.numel() for p in edge_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in edge_model.parameters())
    print(f"Trainable edge gate parameters: {trainable} / {total_params} ({trainable/total_params*100:.4f}%)")
    _tracker.reset_peak()
    _tracker.snap("Edge model loaded")

    gate_params = [p for p in edge_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    edge_model.train()
    total_steps = 0
    start = time.time()
    pbar = tqdm(range(num_epochs), desc="Edge pruning")

    for epoch in pbar:
        ep_loss, ep_kl, ep_sp = 0.0, 0.0, 0.0

        for batch in train_dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()

            edge_out = edge_model(
                input_ids=batch["input_ids"],
                corrupted_input_ids=batch["corrupted_input_ids"],
                attention_mask=batch["attention_mask"],
            )
            with torch.no_grad():
                target_out = full_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            # KL loss at prediction position
            bs = edge_out.logits.size(0)
            total_kl = 0
            for i in range(bs):
                pred_pos = batch["prefix_length"][i].item() - 1
                total_kl += F.kl_div(
                    F.log_softmax(edge_out.logits[i, pred_pos], dim=-1),
                    F.log_softmax(target_out.logits[i, pred_pos], dim=-1),
                    reduction="sum", log_target=True,
                )
            kl_loss = total_kl / bs

            # Task loss
            idx = torch.arange(bs, device=device)
            pred_pos = batch["prefix_length"] - 1
            lg = edge_out.logits[idx, pred_pos, batch["target_token"]]
            lb = edge_out.logits[idx, pred_pos, batch["distractor_token"]]
            task_loss = F.relu(0.1 - (lg - lb)).mean()

            sp_loss = edge_model.get_sparsity_loss(step=total_steps)["total_sparsity"]

            loss = (1 - lambda_sparsity) * (kl_loss + task_loss) + lambda_sparsity * sp_loss
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_kl += kl_loss.item()
            ep_sp += sp_loss.item()
            total_steps += 1

        scheduler.step()
        n = len(train_dataloader)
        pbar.set_postfix(L=f"{ep_loss/n:.3f}", Sp=f"{ep_sp/n:.3f}",
                         LR=f"{scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 10 == 0:
            edge_model.eval()
            run_evaluation(edge_model, f"Edge Ep {epoch+1}", full_model,
                           val_dataloader, device, tokenizer=tokenizer)
            edge_model.train()

        if epoch == 0:
            _tracker.snap("Edge epoch 1 (fwd+bwd+optim)")

    print(f"Edge pruning time: {time.time() - start:.1f}s")
    _tracker.snap("Edge pruning done")

    return edge_model


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Node + Edge circuit pruning for GPT-2 on Gender Pronouns")
    parser.add_argument("--skip-node-pruning", action="store_true",
                        help="Skip node pruning; load active nodes from checkpoint")
    parser.add_argument("--node-checkpoint",
                        default="edge_pruning/active_nodes_gp.json",
                        help="Path to save/load active node specification")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--node-epochs", type=int, default=500)
    parser.add_argument("--edge-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-1)
    parser.add_argument("--node-lambda-sparsity", type=float, default=0.975)
    parser.add_argument("--edge-lambda-sparsity", type=float, default=0.95)
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=10000)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -- GPU info --
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {gpu.name} | {gpu.total_memory / 1024**3:.1f} GB total")

    # -- Tokenizer & full model --
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in full_model.parameters():
        p.requires_grad = False
    _tracker.snap("Full model loaded")

    # -- Dataset --
    print("\nSetting up Gender Pronouns dataset...")
    train_data = load_or_generate_gp_data(split="train", num_samples=args.train_samples)
    val_data = load_or_generate_gp_data(split="validation", num_samples=args.val_samples)
    test_data = load_or_generate_gp_data(split="test", num_samples=args.test_samples)

    val_data = filter_dataset_by_model_correctness(
        val_data, full_model, tokenizer, DEVICE,
        max_length=args.max_seq_length, batch_size=args.batch_size
    )
    test_data = filter_dataset_by_model_correctness(
        test_data, full_model, tokenizer, DEVICE,
        max_length=args.max_seq_length, batch_size=args.batch_size
    )

    train_dl = DataLoader(GPDataset(train_data, tokenizer, max_length=args.max_seq_length),
                          batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(GPDataset(val_data, tokenizer, max_length=args.max_seq_length),
                        batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(GPDataset(test_data, tokenizer, max_length=args.max_seq_length),
                         batch_size=args.batch_size, shuffle=False)

    # -- Baseline --
    print("\n--- Baseline evaluation ---")
    baseline_results = run_evaluation(full_model, "Full Model", None, test_dl, DEVICE, tokenizer=tokenizer)

    # =====================================================================
    # PHASE 1: NODE PRUNING
    # =====================================================================
    node_stats = None
    node_eval_results = None

    if args.skip_node_pruning and os.path.exists(args.node_checkpoint):
        active_heads, active_mlps = load_active_nodes(args.node_checkpoint)
    else:
        node_model, node_stats = run_node_pruning(
            args.model, tokenizer, full_model,
            train_dl, val_dl, test_dl,
            DEVICE, num_epochs=args.node_epochs, lr=args.lr,
            lambda_sparsity=args.node_lambda_sparsity,
        )
        active_heads, active_mlps = extract_active_nodes(node_model)
        save_active_nodes(active_heads, active_mlps, args.node_checkpoint)

        # Capture node-pruned fidelity before deleting
        node_model.eval()
        node_eval_results = run_evaluation(
            node_model, "Node-Pruned Circuit (final)", full_model,
            test_dl, DEVICE, tokenizer=tokenizer
        )

        del node_model
        torch.cuda.empty_cache()
        _tracker.snap("Node model freed")

    # Print active nodes summary
    total_active_heads = sum(len(v) for v in active_heads.values())
    print(f"\nActive nodes: {total_active_heads} heads + {len(active_mlps)} MLPs "
          f"+ 1 embedding = {total_active_heads + len(active_mlps) + 1} total sources")

    # Dense edge count (no edge pruning)
    dense_stats = count_dense_edges(active_heads, active_mlps)

    # =====================================================================
    # PHASE 2: EDGE PRUNING
    # =====================================================================
    edge_model = run_edge_pruning(
        args.model, tokenizer, full_model,
        active_heads, active_mlps,
        train_dl, val_dl, test_dl,
        DEVICE, num_epochs=args.edge_epochs, lr=3e-2,
        lambda_sparsity=args.edge_lambda_sparsity,
    )

    # -- Analysis --
    edge_stats = analyze_edge_circuit(edge_model)

    # -- Final evaluation --
    edge_model.eval()
    edge_eval_results = run_evaluation(
        edge_model, "Edge-Pruned Circuit", full_model,
        test_dl, DEVICE, tokenizer=tokenizer
    )

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    W = 80

    # --- 1. Node Pruning Results ---
    print("\n" + "=" * W)
    print("  PHASE 1 SUMMARY — NODE PRUNING (Gender Pronouns)")
    print("=" * W)

    if node_stats and "granularity_stats" in node_stats:
        gs = node_stats["granularity_stats"]
        print(f"\n{'Component':<25} {'Active':>8} {'Total':>8} {'Pruned %':>10}")
        print("-" * 55)
        for key in ["attention_heads", "attention_blocks", "attention_neurons",
                     "mlp_blocks", "mlp_hidden", "mlp_output"]:
            s = gs.get(key, {"total": 0, "active": 0})
            if s["total"] > 0:
                pct = (s["total"] - s["active"]) / s["total"] * 100
                print(f"  {key:<23} {s['active']:>8,} {s['total']:>8,} {pct:>9.1f}%")

    if node_stats and "prunable_compression" in node_stats:
        pc = node_stats["prunable_compression"]
        print(f"\n  Parameter reduction:     {pc['reduction_percentage']:.1f}%")
        print(f"  Compression ratio:       {pc['compression_ratio']:.2f}x")
        print(f"  Effective compression:   {pc['effective_compression']:.2f}x")
        print(f"  Active / Total prunable: {pc['active_prunable_params']:,} / {pc['total_prunable_params']:,}")

    if node_eval_results:
        print(f"\n  Fidelity Metrics (Node-Pruned):")
        print(f"    Accuracy:         {node_eval_results['accuracy']:.4f}  (baseline: {baseline_results['accuracy']:.4f}, drop: {baseline_results['accuracy'] - node_eval_results['accuracy']:+.4f})")
        print(f"    Logit Difference: {node_eval_results['logit_diff']:.4f}  (baseline: {baseline_results['logit_diff']:.4f})")
        print(f"    KL Divergence:    {node_eval_results['kl_div']:.4f}")
        print(f"    Exact Match:      {node_eval_results['exact_match']:.4f}")

    # --- 2. Dense Edge Count ---
    print("\n" + "=" * W)
    print("  DENSE EDGES (between surviving nodes, before edge pruning)")
    print("=" * W)
    full_e = dense_stats["full_total"]
    dense_e = dense_stats["dense_total"]
    print(f"\n  Full model edges:    {full_e:>10,}")
    print(f"  Dense edges:         {dense_e:>10,}  ({dense_e / full_e:.2%} of full)")
    print(f"  Node-level reduction:{full_e - dense_e:>10,}  ({(full_e - dense_e) / full_e:.2%})")

    print(f"\n  {'Category':<12} {'Edges':>10}")
    print(f"  {'-'*24}")
    print(f"  {'Output':<12} {dense_stats['dense_output']:>10,}")
    print(f"  {'MLP input':<12} {dense_stats['dense_mlp']:>10,}")
    print(f"  {'Q input':<12} {dense_stats['dense_q']:>10,}")
    print(f"  {'K input':<12} {dense_stats['dense_k']:>10,}")
    print(f"  {'V input':<12} {dense_stats['dense_v']:>10,}")

    # --- 3. Edge Pruning Results ---
    te = edge_stats["total_edges"]
    ae = edge_stats["active_edges"]

    print("\n" + "=" * W)
    print("  PHASE 2 SUMMARY — EDGE PRUNING (Gender Pronouns)")
    print("=" * W)

    print(f"\n  {'Category':<18} {'Active':>8} {'Total':>8} {'Kept %':>9} {'Pruned %':>10}")
    print(f"  {'-'*57}")
    for cat_key, cat_data in edge_stats["stats"].items():
        name = cat_key.replace("_", " ").title()
        t, a = cat_data["total"], cat_data["active"]
        if t > 0:
            kept = a / t * 100
            pruned = 100 - kept
            print(f"  {name:<18} {a:>8,} {t:>8,} {kept:>8.1f}% {pruned:>9.1f}%")
    print(f"  {'-'*57}")
    if te > 0:
        kept_all = ae / te * 100
        pruned_all = 100 - kept_all
        print(f"  {'TOTAL':<18} {ae:>8,} {te:>8,} {kept_all:>8.1f}% {pruned_all:>9.1f}%")
    if ae > 0:
        print(f"\n  Edge compression: {te / ae:.2f}x")
    print(f"  vs full model:    {ae:,} / {full_e:,} ({ae / full_e:.2%} of all possible edges)")

    # --- 4. Fidelity comparison table ---
    print("\n" + "=" * W)
    print("  FIDELITY COMPARISON")
    print("=" * W)

    header = f"  {'Model':<28} {'Accuracy':>10} {'Logit Diff':>12} {'KL Div':>10} {'Exact Match':>13}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    print(f"  {'Baseline (Full Model)':<28} {baseline_results['accuracy']:>10.4f} {baseline_results['logit_diff']:>12.4f} {'—':>10} {'—':>13}")

    if node_eval_results:
        print(f"  {'Node-Pruned Circuit':<28} {node_eval_results['accuracy']:>10.4f} {node_eval_results['logit_diff']:>12.4f} {node_eval_results['kl_div']:>10.4f} {node_eval_results['exact_match']:>13.4f}")

    print(f"  {'Edge-Pruned Circuit':<28} {edge_eval_results['accuracy']:>10.4f} {edge_eval_results['logit_diff']:>12.4f} {edge_eval_results['kl_div']:>10.4f} {edge_eval_results['exact_match']:>13.4f}")

    # Deltas
    if node_eval_results:
        print(f"\n  {'Delta (Edge vs Node)':<28} "
              f"{edge_eval_results['accuracy'] - node_eval_results['accuracy']:>+10.4f} "
              f"{edge_eval_results['logit_diff'] - node_eval_results['logit_diff']:>+12.4f} "
              f"{edge_eval_results['kl_div'] - node_eval_results['kl_div']:>+10.4f} "
              f"{edge_eval_results['exact_match'] - node_eval_results['exact_match']:>+13.4f}")

    print(f"  {'Delta (Edge vs Baseline)':<28} "
          f"{edge_eval_results['accuracy'] - baseline_results['accuracy']:>+10.4f} "
          f"{edge_eval_results['logit_diff'] - baseline_results['logit_diff']:>+12.4f} "
          f"{'—':>10} {'—':>13}")

    # --- 5. Overall compression pipeline ---
    print("\n" + "=" * W)
    print("  END-TO-END COMPRESSION PIPELINE")
    print("=" * W)

    print(f"\n  {'Stage':<35} {'Edges':>10} {'% of Full':>10} {'Cumulative Reduction':>22}")
    print(f"  {'-'*79}")
    print(f"  {'Full model (all edges)':<35} {full_e:>10,} {'100.0%':>10} {'—':>22}")
    print(f"  {'After node pruning (dense)':<35} {dense_e:>10,} {dense_e/full_e*100:>9.1f}% {(full_e-dense_e)/full_e*100:>21.1f}%")
    print(f"  {'After edge pruning (final)':<35} {ae:>10,} {ae/full_e*100:>9.1f}% {(full_e-ae)/full_e*100:>21.1f}%")

    if ae > 0:
        print(f"\n  Overall edge compression: {full_e / ae:.2f}x")

    print("=" * W)

    # -- GPU memory map --
    _tracker.print_report()


if __name__ == "__main__":
    main()
