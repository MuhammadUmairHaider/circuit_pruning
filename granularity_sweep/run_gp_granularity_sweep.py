"""
Granularity Ablation Sweep for Gender Pronouns (GP) task.

For each granularity (attention_heads, mlp_hidden, mlp_output, attention_neurons,
attention_blocks, mlp_blocks), runs a full training with ONLY that granularity
enabled, plus an "all" baseline with everything enabled.

Fixed: task=GP, lambda_sparsity=0.70

Outputs:
  - granularity_sweep/results_gp/granularity_results.csv
  - granularity_sweep/results_gp/granularity_results.json
  - granularity_sweep/results_gp/per_epoch/

Usage:
  python -m granularity_sweep.run_gp_granularity_sweep
  python -m granularity_sweep.run_gp_granularity_sweep --granularities attention_heads mlp_blocks
"""

import sys
import os
import argparse
import json
import csv
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_circuit import (
    PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2,
    GPT2LMHeadModel,
    PruningConfig,
)
from utils import disable_dropout, analyze_and_finalize_circuit

# =============================================================================
# GRANULARITY DEFINITIONS
# =============================================================================

ALL_GRANULARITIES = [
    "attention_heads",
    "mlp_hidden",
    "mlp_output",
    "attention_neurons",
    "attention_blocks",
    "mlp_blocks",
]

GRANULARITY_LABELS = {
    "attention_heads": "Attn Heads",
    "mlp_hidden": "MLP Hidden",
    "mlp_output": "MLP Output",
    "attention_neurons": "Attn Neurons",
    "attention_blocks": "Attn Blocks",
    "mlp_blocks": "MLP Blocks",
    "all": "All (combined)",
}

# GP training config (mirrors gp.py)
GP_CONFIG = {
    "num_epochs": 500,
    "learning_rate": 3e-1,
    "batch_size": 64,
    "max_seq_len": 32,
}

LAMBDA_SPARSITY = 0.975


def build_pruning_config(granularity: str) -> PruningConfig:
    """
    Build a PruningConfig with only the specified granularity enabled.
    If granularity == "all", enable everything.
    """
    cfg = PruningConfig(
        init_value=0.5,
        sparsity_warmup_steps=1000,
        depth_penalty_scaling=0.0,
        prune_attention_heads=False,
        lambda_attention_heads=1.0,
        prune_mlp_hidden=False,
        lambda_mlp_hidden=1.0,
        prune_mlp_output=False,
        lambda_mlp_output=1.0,
        prune_attention_neurons=False,
        lambda_attention_neurons=1.0,
        prune_attention_blocks=False,
        lambda_attention_blocks=1.0,
        prune_mlp_blocks=False,
        lambda_mlp_blocks=2.0,
        prune_full_layers=False,
        lambda_full_layers=0.0,
        prune_embedding=False,
        lambda_embedding=1.0,
    )

    if granularity == "all":
        cfg.prune_attention_heads = True
        cfg.prune_mlp_hidden = True
        cfg.prune_mlp_output = True
        cfg.prune_attention_neurons = True
        cfg.prune_attention_blocks = True
        cfg.prune_mlp_blocks = True
    elif granularity == "attention_heads":
        cfg.prune_attention_heads = True
    elif granularity == "mlp_hidden":
        cfg.prune_mlp_hidden = True
    elif granularity == "mlp_output":
        cfg.prune_mlp_output = True
    elif granularity == "attention_neurons":
        cfg.prune_attention_neurons = True
    elif granularity == "attention_blocks":
        cfg.prune_attention_blocks = True
    elif granularity == "mlp_blocks":
        cfg.prune_mlp_blocks = True
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    return cfg


# =============================================================================
# DATA LOADING (GP)
# =============================================================================

def load_gp_data(tokenizer, full_model, device, batch_size, max_seq_len):
    from dataset.gp import (
        GPDataset,
        load_or_generate_gp_data,
        run_evaluation,
        filter_dataset_by_model_correctness,
    )

    train_data = load_or_generate_gp_data(split="train", num_samples=100000)
    val_data = load_or_generate_gp_data(split="validation", num_samples=10000)
    test_data = load_or_generate_gp_data(split="test", num_samples=100000)

    val_data = filter_dataset_by_model_correctness(
        val_data, full_model, tokenizer, device, max_length=max_seq_len, batch_size=batch_size
    )
    test_data = filter_dataset_by_model_correctness(
        test_data, full_model, tokenizer, device, max_length=max_seq_len, batch_size=batch_size
    )

    train_loader = DataLoader(GPDataset(train_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GPDataset(val_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GPDataset(test_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, run_evaluation


# =============================================================================
# GP TRAINING STEP (mirrors gp.py)
# =============================================================================

def train_step_gp(circuit_model, full_model, batch, device):
    circuit_outputs = circuit_model(
        input_ids=batch["input_ids"],
        corrupted_input_ids=batch["corrupted_input_ids"],
        attention_mask=batch["attention_mask"],
    )

    with torch.no_grad():
        target_outputs = full_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    batch_size_curr = circuit_outputs.logits.size(0)
    total_kl = 0

    for i in range(batch_size_curr):
        pred_pos = batch["prefix_length"][i] - 1

        circuit_logits = circuit_outputs.logits[i, pred_pos, :]
        target_logits = target_outputs.logits[i, pred_pos, :]

        kl = F.kl_div(
            F.log_softmax(circuit_logits, dim=-1),
            F.log_softmax(target_logits, dim=-1),
            reduction="sum",
            log_target=True,
        )
        total_kl += kl

    logit_good = circuit_outputs.logits[
        torch.arange(batch_size_curr, device=device),
        batch["prefix_length"] - 1,
        batch["target_token"],
    ]
    logit_bad = circuit_outputs.logits[
        torch.arange(batch_size_curr, device=device),
        batch["prefix_length"] - 1,
        batch["distractor_token"],
    ]
    task_loss = F.relu(0.1 - (logit_good - logit_bad)).mean()

    kl_loss = total_kl / batch_size_curr
    return kl_loss, task_loss


# =============================================================================
# PRUNING STATISTICS
# =============================================================================

def extract_pruning_stats(circuit_model, total_steps):
    circuit_model.eval()
    sparsity_info = circuit_model.get_sparsity_loss(step=total_steps)
    sparsity_components = {k: v.item() if torch.is_tensor(v) else v for k, v in sparsity_info.items()}

    finalize_results = analyze_and_finalize_circuit(circuit_model, verbose=False)
    granularity = finalize_results["granularity_stats"]
    prunable = finalize_results["prunable_compression"]

    STANDARD_GRANULARITIES = [
        "attention_heads", "attention_blocks", "attention_neurons",
        "mlp_blocks", "mlp_hidden", "mlp_output",
    ]
    pruning_pcts = {}
    for key in STANDARD_GRANULARITIES:
        stats = granularity.get(key, {"total": 0, "active": 0})
        if stats["total"] > 0:
            pruning_pcts[f"pruned_pct_{key}"] = (stats["total"] - stats["active"]) / stats["total"] * 100
            pruning_pcts[f"active_{key}"] = stats["active"]
            pruning_pcts[f"total_{key}"] = stats["total"]
        else:
            pruning_pcts[f"pruned_pct_{key}"] = 0.0
            pruning_pcts[f"active_{key}"] = 0
            pruning_pcts[f"total_{key}"] = 0

    return {
        "sparsity_components": sparsity_components,
        "pruning_percentages": pruning_pcts,
        "compression_ratio": prunable["compression_ratio"],
        "reduction_percentage": prunable["reduction_percentage"],
        "active_prunable_params": prunable["active_prunable_params"],
        "total_prunable_params": prunable["total_prunable_params"],
        "effective_compression": prunable["effective_compression"],
    }


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_single_experiment(granularity: str, device: str, data_cache: dict, seed: int = 42):
    torch.manual_seed(seed)

    print(f"\n{'='*80}")
    print(f"  GRANULARITY ABLATION (GP): {GRANULARITY_LABELS.get(granularity, granularity)}")
    print(f"  lambda_sparsity={LAMBDA_SPARSITY}, task=GP")
    print(f"{'='*80}")

    tokenizer = data_cache["tokenizer"]
    full_model = data_cache["full_model"]
    train_loader = data_cache["train_loader"]
    val_loader = data_cache["val_loader"]
    test_loader = data_cache["test_loader"]
    run_evaluation = data_cache["run_evaluation"]
    baseline_results = data_cache["baseline_results"]

    pruning_config = build_pruning_config(granularity)

    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning("gpt2", pruning_config).to(device).eval()
    disable_dropout(circuit_model)

    for name, param in circuit_model.named_parameters():
        param.requires_grad = "gate" in name

    trainable = sum(p.numel() for p in circuit_model.parameters() if p.requires_grad)
    print(f"  Trainable gate parameters: {trainable}")

    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=GP_CONFIG["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=GP_CONFIG["num_epochs"], eta_min=1e-4)

    circuit_model.train()
    total_steps = 0
    epoch_records = []

    start_time = time.time()
    epoch_pbar = tqdm(range(GP_CONFIG["num_epochs"]), desc=f"gran={granularity}")

    for epoch in epoch_pbar:
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_sparsity = 0.0
        epoch_task_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            kl_loss, task_loss = train_step_gp(circuit_model, full_model, batch, device)
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)["total_sparsity"]

            loss = (1 - LAMBDA_SPARSITY) * (kl_loss + task_loss) + LAMBDA_SPARSITY * sparsity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_sparsity += sparsity_loss.item()
            epoch_task_loss += task_loss.item()
            total_steps += 1

        scheduler.step()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_kl = epoch_kl / n_batches
        avg_sparsity = epoch_sparsity / n_batches
        avg_task_loss = epoch_task_loss / n_batches

        epoch_pbar.set_postfix({"L": f"{avg_loss:.3f}", "KL": f"{avg_kl:.3f}", "Sp": f"{avg_sparsity:.3f}"})

        epoch_record = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_kl_loss": avg_kl,
            "avg_sparsity_loss": avg_sparsity,
            "avg_task_loss": avg_task_loss,
        }

        if (epoch + 1) % 50 == 0:
            circuit_model.eval()
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Val Ep {epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_loader,
                device=device,
                tokenizer=tokenizer,
            )
            epoch_record["val_accuracy"] = val_results.get("accuracy", 0.0)
            epoch_record["val_kl_div"] = val_results.get("kl_div", 0.0)
            epoch_record["val_logit_diff"] = val_results.get("logit_diff", 0.0)
            circuit_model.train()

        epoch_records.append(epoch_record)

    training_time = time.time() - start_time

    # Final evaluation
    circuit_model.eval()
    pruning_stats = extract_pruning_stats(circuit_model, total_steps)

    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name=f"Final gran={granularity}",
        full_model_for_faithfulness=full_model,
        dataloader=test_loader,
        device=device,
        tokenizer=tokenizer,
    )

    result = {
        "granularity": granularity,
        "granularity_label": GRANULARITY_LABELS.get(granularity, granularity),
        "lambda_sparsity": LAMBDA_SPARSITY,
        "task": "gp",
        "seed": seed,
        "num_epochs": GP_CONFIG["num_epochs"],
        "training_time_seconds": training_time,
        "trainable_gate_params": trainable,
        # Baseline
        "baseline_accuracy": baseline_results["accuracy"],
        "baseline_logit_diff": baseline_results["logit_diff"],
        # Final test
        "final_accuracy": final_results.get("accuracy", 0.0),
        "final_logit_diff": final_results.get("logit_diff", 0.0),
        "final_kl_div": final_results.get("kl_div", 0.0),
        "final_exact_match": final_results.get("exact_match", 0.0),
        "accuracy_drop": baseline_results["accuracy"] - final_results.get("accuracy", 0.0),
        # Training
        "final_train_loss": epoch_records[-1]["avg_loss"],
        "final_train_kl": epoch_records[-1]["avg_kl_loss"],
        "final_train_sparsity": epoch_records[-1]["avg_sparsity_loss"],
        # Pruning
        "compression_ratio": pruning_stats["compression_ratio"],
        "reduction_percentage": pruning_stats["reduction_percentage"],
        "active_prunable_params": pruning_stats["active_prunable_params"],
        "total_prunable_params": pruning_stats["total_prunable_params"],
        "effective_compression": pruning_stats["effective_compression"],
        **pruning_stats["pruning_percentages"],
        **{f"sparsity_{k}": v for k, v in pruning_stats["sparsity_components"].items()},
    }

    return result, epoch_records


# =============================================================================
# CSV HELPERS
# =============================================================================

def write_csv(results, filepath):
    if not results:
        return
    all_keys = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Saved CSV: {filepath}")


def write_epoch_csv(granularity, epoch_records, dirpath):
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, f"gp_{granularity}_epochs.csv")
    if not epoch_records:
        return
    keys = list(epoch_records[0].keys())
    for rec in epoch_records:
        for k in rec.keys():
            if k not in keys:
                keys.append(k)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for rec in epoch_records:
            writer.writerow(rec)
    print(f"  Saved epoch CSV: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Granularity ablation sweep on GP")
    parser.add_argument(
        "--granularities",
        nargs="+",
        default=["all"] + ALL_GRANULARITIES,
        help="Granularities to test (default: all + each individual)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gp"),
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    epoch_dir = os.path.join(args.output_dir, "per_epoch")

    print(f"Device: {device}")
    print(f"Granularities to test: {args.granularities}")
    print(f"Lambda sparsity: {LAMBDA_SPARSITY}")

    # Load data ONCE (shared across all runs)
    print("\n--- Loading GP data (shared across all granularity runs) ---")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    train_loader, val_loader, test_loader, run_evaluation = load_gp_data(
        tokenizer, full_model, device, GP_CONFIG["batch_size"], GP_CONFIG["max_seq_len"]
    )

    # Baseline evaluation (once)
    print("\n--- Baseline evaluation ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline Full Model",
        full_model_for_faithfulness=None,
        dataloader=test_loader,
        device=device,
        tokenizer=tokenizer,
    )
    print(f"  Baseline accuracy: {baseline_results.get('accuracy', 0.0):.4f}")

    data_cache = {
        "tokenizer": tokenizer,
        "full_model": full_model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "run_evaluation": run_evaluation,
        "baseline_results": {
            "accuracy": baseline_results.get("accuracy", 0.0),
            "logit_diff": baseline_results.get("logit_diff", 0.0),
        },
    }

    all_results = []

    for gran in args.granularities:
        try:
            result, epoch_records = run_single_experiment(
                granularity=gran,
                device=device,
                data_cache=data_cache,
                seed=args.seed,
            )
            all_results.append(result)

            write_epoch_csv(gran, epoch_records, epoch_dir)

            # Save running results
            write_csv(all_results, os.path.join(args.output_dir, "granularity_results.csv"))
            with open(os.path.join(args.output_dir, "granularity_results.json"), "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            print(f"\n  SUMMARY: {GRANULARITY_LABELS.get(gran, gran)}")
            print(f"    Accuracy: {result['final_accuracy']:.4f} (drop: {result['accuracy_drop']:.4f})")
            print(f"    KL Div: {result['final_kl_div']:.4f}")
            print(f"    Compression: {result['compression_ratio']:.2f}x ({result['reduction_percentage']:.1f}% pruned)")

        except Exception as e:
            print(f"\n  ERROR: granularity={gran}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print(f"\n{'='*140}")
    print("GRANULARITY ABLATION (GP) COMPLETE - SUMMARY")
    print(f"{'='*140}")
    header = (
        f"{'Granularity':<18} {'Acc':<8} {'LogitDiff':<10} {'KL':<8} "
        f"{'Attn Blk%':<10} {'MLP Blk%':<10} {'Attn Hd%':<9} "
        f"{'Attn Neur%':<11} {'MLP Hid%':<9} {'MLP Out%':<9} {'Time(s)':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['granularity_label']:<18} "
            f"{r['final_accuracy']:<8.4f} {r['final_logit_diff']:<10.4f} {r['final_kl_div']:<8.4f} "
            f"{r.get('pruned_pct_attention_blocks', 0.0):<10.1f}"
            f"{r.get('pruned_pct_mlp_blocks', 0.0):<10.1f}"
            f"{r.get('pruned_pct_attention_heads', 0.0):<9.1f}"
            f" {r.get('pruned_pct_attention_neurons', 0.0):<11.1f}"
            f"{r.get('pruned_pct_mlp_hidden', 0.0):<9.1f}"
            f" {r.get('pruned_pct_mlp_output', 0.0):<9.1f}"
            f"{r['training_time_seconds']:<8.1f}"
        )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
