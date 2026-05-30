"""
Lambda sparsity hyperparameter sweep for Llama models on IOI and GP tasks.

Bare-bones GPT-2-style training (no adaptive scheduler), sweeping lambda_sparsity.
All prunable granularities have weight 1.0. 500 epochs per run.

Models: meta-llama/Llama-3.2-1B, meta-llama/Llama-3.1-8B
Tasks:  IOI, GP

Usage:
  # Full sweep (2 models x 2 tasks x N lambdas)
  python -m llama_sweep.run_llama_sweep

  # Single model, single task
  python -m llama_sweep.run_llama_sweep --models meta-llama/Llama-3.2-1B --tasks ioi --lambdas 0.70 0.80 0.90

  # With flash attention and custom HF token
  python -m llama_sweep.run_llama_sweep --flash-attn --hf-token YOUR_TOKEN

  # Dry run (2 epochs, tiny data)
  python -m llama_sweep.run_llama_sweep --dry-run
"""

import sys
import os
import argparse
import json
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from utils import disable_dropout, analyze_and_finalize_circuit


# =============================================================================
# PRUNING CONFIG: all granularities at weight 1.0 (GPT-2 style)
# =============================================================================

@dataclass
class LlamaSweepPruningConfig(PruningConfig):
    init_value: float = 0.5
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.0

    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0

    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0

    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0

    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0

    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 1.0

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0

    prune_embedding: bool = False
    lambda_embedding: float = 1.0


# =============================================================================
# TASK CONFIGS
# =============================================================================

TASK_CONFIGS = {
    "ioi": {
        "num_epochs": 500,
        "learning_rate": 1e-2,
        "batch_size": 32,
    },
    "gp": {
        "num_epochs": 500,
        "learning_rate": 1e-2,
        "batch_size": 32,
    },
    "copycolors": {
        "num_epochs": 500,
        "learning_rate": 1e-2,
        "batch_size": 16,
        "max_seq_len": 128,
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_task_data(task: str, tokenizer, full_model, device: str, cfg: dict, dry_run: bool = False):
    """Load and filter data. Returns (train_loader, val_loader, test_loader, eval_fn)."""
    batch_size = cfg["batch_size"]

    if dry_run:
        n_train, n_val, n_test = 10, 5, 5
    else:
        n_train, n_val, n_test = 200, 200, 1000

    if task == "ioi":
        from dataset.ioi_llama import (
            IOIDatasetLlama,
            generate_ioi_data_llama,
            run_evaluation,
            filter_dataset_by_model_correctness,
        )

        train_data = generate_ioi_data_llama(n_train, tokenizer, seed=42)
        val_data = generate_ioi_data_llama(n_val, tokenizer, seed=123)
        test_data = generate_ioi_data_llama(n_test, tokenizer, seed=456)

        val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device, batch_size=batch_size)
        test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, device, batch_size=batch_size)

        train_loader = DataLoader(IOIDatasetLlama(train_data, tokenizer), batch_size=batch_size, shuffle=False, pin_memory=True)
        val_loader = DataLoader(IOIDatasetLlama(val_data, tokenizer), batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(IOIDatasetLlama(test_data, tokenizer), batch_size=batch_size, pin_memory=True)

        return train_loader, val_loader, test_loader, run_evaluation

    elif task == "gp":
        from dataset.gp_llama import (
            GPDatasetLlama,
            load_or_generate_gp_data,
            run_evaluation,
            filter_dataset_by_model_correctness,
        )

        train_data = load_or_generate_gp_data(split="train", num_samples=n_train)
        val_data = load_or_generate_gp_data(split="validation", num_samples=n_val)
        test_data = load_or_generate_gp_data(split="test", num_samples=n_test)

        val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device, batch_size=batch_size)
        test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, device, batch_size=batch_size)

        train_loader = DataLoader(GPDatasetLlama(train_data, tokenizer), batch_size=batch_size, shuffle=False, pin_memory=True)
        val_loader = DataLoader(GPDatasetLlama(val_data, tokenizer), batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(GPDatasetLlama(test_data, tokenizer), batch_size=batch_size, pin_memory=True)

        return train_loader, val_loader, test_loader, run_evaluation

    elif task == "copycolors":
        from dataset.copycolors_llama import (
            CopyColorsDatasetLlama,
            load_copycolors_data,
            run_evaluation,
            filter_dataset_by_model_correctness,
        )

        max_seq_len = cfg.get("max_seq_len", 128)

        train_data = load_copycolors_data(split="train", num_samples=n_train)
        val_data = load_copycolors_data(split="validation", num_samples=n_val)
        test_data = load_copycolors_data(split="test", num_samples=n_test)

        val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device, batch_size=batch_size)
        test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, device, batch_size=batch_size)

        train_loader = DataLoader(CopyColorsDatasetLlama(train_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(CopyColorsDatasetLlama(val_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(CopyColorsDatasetLlama(test_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, pin_memory=True)

        return train_loader, val_loader, test_loader, run_evaluation

    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# TRAINING STEPS
# =============================================================================

def train_step_ioi(circuit_model, cached_logits, batch_idx, batch, device):
    """IOI training step for Llama. Uses cached full-model logits."""
    outputs = circuit_model(
        input_ids=batch["input_ids"],
        corrupted_input_ids=batch["corrupted_input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )

    batch_size = outputs.logits.size(0)

    # KL loss at target positions
    total_kl = 0
    for i in range(batch_size):
        t_start = batch["T_Start"][i].item() - 1
        t_end = batch["T_End"][i].item() - 1
        valid_length = batch["attention_mask"][i].sum().item()
        end_pos = min(t_end, valid_length)

        if t_start < end_pos:
            kl = F.kl_div(
                F.log_softmax(outputs.logits[i, t_start:end_pos, :].float(), dim=-1),
                F.log_softmax(cached_logits[batch_idx][i, t_start:end_pos, :].float(), dim=-1),
                reduction="sum",
                log_target=True,
            )
            total_kl += kl

    kl_loss = total_kl / batch_size

    # Task loss
    pos_good = batch["T_Start"] - 1
    token_good = batch["target_tokens"][:, 0]
    pos_bad = batch["D_Start"] - 1
    token_bad = batch["distractor_tokens"][:, 0]
    batch_indices = torch.arange(batch_size, device=device)

    logit_good = outputs.logits[batch_indices, pos_good, token_good].float()
    logit_bad = outputs.logits[batch_indices, pos_bad, token_bad].float()
    task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()

    return kl_loss, task_loss


def train_step_gp(circuit_model, cached_logits, batch_idx, batch, device):
    """GP training step for Llama. Uses cached full-model logits."""
    outputs = circuit_model(
        input_ids=batch["input_ids"],
        corrupted_input_ids=batch["corrupted_input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )

    batch_size = outputs.logits.size(0)

    # KL loss at prediction position
    total_kl = 0
    for i in range(batch_size):
        pred_pos = batch["prefix_length"][i].item() - 1
        valid_length = batch["attention_mask"][i].sum().item()

        if pred_pos < valid_length:
            kl = F.kl_div(
                F.log_softmax(outputs.logits[i, pred_pos].float(), dim=-1),
                F.log_softmax(cached_logits[batch_idx][i, pred_pos].float(), dim=-1),
                reduction="sum",
                log_target=True,
            )
            total_kl += kl

    kl_loss = total_kl / batch_size

    # Task loss
    logit_target = outputs.logits[
        torch.arange(batch_size, device=device),
        batch["prefix_length"] - 1,
        batch["target_token"],
    ].float()
    logit_distractor = outputs.logits[
        torch.arange(batch_size, device=device),
        batch["prefix_length"] - 1,
        batch["distractor_token"],
    ].float()
    task_loss = F.relu(4.0 - (logit_target - logit_distractor)).mean()

    return kl_loss, task_loss


def train_step_copycolors(circuit_model, cached_logits, batch_idx, batch, device):
    """CopyColors MCQA training step for Llama. Uses cached full-model logits."""
    outputs = circuit_model(
        input_ids=batch["input_ids"],
        corrupted_input_ids=batch["corrupted_input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )

    batch_size = outputs.logits.size(0)

    # KL loss at prediction position
    total_kl = 0
    for i in range(batch_size):
        pred_pos = batch["prefix_length"][i].item() - 1
        valid_length = batch["attention_mask"][i].sum().item()

        if pred_pos < valid_length and pred_pos < outputs.logits.size(1):
            kl = F.kl_div(
                F.log_softmax(outputs.logits[i, pred_pos].float(), dim=-1),
                F.log_softmax(cached_logits[batch_idx][i, pred_pos].float(), dim=-1),
                reduction="sum",
                log_target=True,
            )
            total_kl += kl

    kl_loss = total_kl / batch_size

    # Task loss: correct answer letter should beat distractor
    batch_indices = torch.arange(batch_size, device=device)
    pred_positions = batch["prefix_length"] - 1

    logit_target = outputs.logits[
        batch_indices, pred_positions, batch["target_token"]
    ].float()
    logit_distractor = outputs.logits[
        batch_indices, pred_positions, batch["distractor_token"]
    ].float()
    task_loss = F.relu(4.0 - (logit_target - logit_distractor)).mean()

    return kl_loss, task_loss


# =============================================================================
# EVALUATION WRAPPER
# =============================================================================

def run_task_evaluation(task, eval_fn, model, full_model, dataloader, device, tokenizer, name=""):
    results = eval_fn(
        model_to_eval=model,
        model_name=name,
        full_model_for_faithfulness=full_model,
        dataloader=dataloader,
        device=device,
        verbose=False,
        tokenizer=tokenizer,
    )
    return {
        "accuracy": results.get("accuracy", 0.0),
        "logit_diff": results.get("logit_diff", 0.0),
        "kl_div": results.get("kl_div", 0.0),
        "exact_match": results.get("exact_match", 0.0),
    }


# =============================================================================
# PRUNING STATS
# =============================================================================

def extract_pruning_stats(circuit_model, total_steps):
    circuit_model.eval()
    sparsity_info = circuit_model.get_sparsity_loss(step=total_steps)
    sparsity_components = {k: v.item() if torch.is_tensor(v) else v for k, v in sparsity_info.items()}

    finalize_results = analyze_and_finalize_circuit(circuit_model, verbose=False)
    granularity = finalize_results["granularity_stats"]
    prunable = finalize_results["prunable_compression"]

    pruning_pcts = {}
    for key, stats in granularity.items():
        if stats["total"] > 0:
            pruning_pcts[f"pruned_pct_{key}"] = (stats["total"] - stats["active"]) / stats["total"] * 100
            pruning_pcts[f"active_{key}"] = stats["active"]
            pruning_pcts[f"total_{key}"] = stats["total"]

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

def run_single_experiment(
    task: str,
    model_name: str,
    lambda_sparsity: float,
    device: str,
    hf_token: Optional[str],
    flash_attn: bool,
    dry_run: bool = False,
    seed: int = 42,
):
    torch.manual_seed(seed)

    cfg = TASK_CONFIGS[task]
    num_epochs = 2 if dry_run else cfg["num_epochs"]

    model_short = model_name.split("/")[-1]
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT: model={model_short}, task={task}, lambda_sparsity={lambda_sparsity:.2f}")
    print(f"{'='*80}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Models ---
    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}
    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    pruning_config = LlamaSweepPruningConfig()

    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        model_name, pruning_config, **model_kwargs
    ).to(device).eval()

    full_model = LlamaForCausalLM.from_pretrained(model_name, **model_kwargs).to(device).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    disable_dropout(circuit_model)

    # Freeze base, unfreeze gates (cast gate params to float32 for stability)
    GATE_PATTERNS = ("_gates.", "_gate.", "embedding_gate.", "layer_gates.")
    for name, param in circuit_model.named_parameters():
        is_gate = any(p in name for p in GATE_PATTERNS)
        param.requires_grad = is_gate
        if is_gate:
            param.data = param.data.float()

    trainable = sum(p.numel() for p in circuit_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in circuit_model.parameters())
    print(f"  Trainable gate params: {trainable} / {total} ({trainable/total*100:.4f}%)")

    # --- Data ---
    train_loader, val_loader, test_loader, eval_fn = load_task_data(
        task, tokenizer, full_model, device, cfg, dry_run=dry_run
    )

    # --- Baseline ---
    print("  Running baseline evaluation...")
    baseline_results = run_task_evaluation(
        task, eval_fn, full_model, None, test_loader, device, tokenizer, name="Baseline"
    )
    print(f"  Baseline accuracy: {baseline_results['accuracy']:.4f}")

    # --- Cache full model outputs ---
    print("  Caching full model outputs...")
    cached_logits = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = full_model(input_ids=batch_gpu["input_ids"], attention_mask=batch_gpu["attention_mask"], use_cache=False)
            cached_logits[batch_idx] = out.logits.detach()

    # --- Select train step function ---
    if task == "ioi":
        train_step_fn = lambda batch_idx, batch: train_step_ioi(circuit_model, cached_logits, batch_idx, batch, device)
    elif task == "gp":
        train_step_fn = lambda batch_idx, batch: train_step_gp(circuit_model, cached_logits, batch_idx, batch, device)
    elif task == "copycolors":
        train_step_fn = lambda batch_idx, batch: train_step_copycolors(circuit_model, cached_logits, batch_idx, batch, device)

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=cfg["learning_rate"])

    circuit_model.train()
    total_steps = 0
    epoch_records = []

    start_time = time.time()
    epoch_pbar = tqdm(range(num_epochs), desc=f"{model_short}/{task}/ls={lambda_sparsity:.2f}")

    for epoch in epoch_pbar:
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_sparsity = 0.0
        epoch_task_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()

            kl_loss, task_loss = train_step_fn(batch_idx, batch)
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)["total_sparsity"]

            loss = (1 - lambda_sparsity) * (kl_loss + task_loss) + lambda_sparsity * sparsity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_sparsity += sparsity_loss.item()
            epoch_task_loss += task_loss.item()
            total_steps += 1

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
            val_results = run_task_evaluation(
                task, eval_fn, circuit_model, full_model, val_loader, device, tokenizer,
                name=f"Val Ep {epoch+1}",
            )
            epoch_record["val_accuracy"] = val_results["accuracy"]
            epoch_record["val_kl_div"] = val_results["kl_div"]
            epoch_record["val_logit_diff"] = val_results["logit_diff"]
            circuit_model.train()

        epoch_records.append(epoch_record)

    training_time = time.time() - start_time

    # --- Final evaluation ---
    print("  Running final evaluation...")
    circuit_model.eval()
    pruning_stats = extract_pruning_stats(circuit_model, total_steps)

    final_results = run_task_evaluation(
        task, eval_fn, circuit_model, full_model, test_loader, device, tokenizer,
        name=f"Final {model_short}/{task}/ls={lambda_sparsity:.2f}",
    )

    # --- Compile ---
    result = {
        "model": model_name,
        "model_short": model_short,
        "task": task,
        "lambda_sparsity": lambda_sparsity,
        "seed": seed,
        "num_epochs": num_epochs,
        "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "training_time_seconds": training_time,
        # Baseline
        "baseline_accuracy": baseline_results["accuracy"],
        "baseline_logit_diff": baseline_results["logit_diff"],
        # Final
        "final_accuracy": final_results["accuracy"],
        "final_logit_diff": final_results["logit_diff"],
        "final_kl_div": final_results["kl_div"],
        "final_exact_match": final_results["exact_match"],
        "accuracy_drop": baseline_results["accuracy"] - final_results["accuracy"],
        # Training
        "final_train_loss": epoch_records[-1]["avg_loss"],
        "final_train_kl": epoch_records[-1]["avg_kl_loss"],
        "final_train_sparsity": epoch_records[-1]["avg_sparsity_loss"],
        "final_train_task_loss": epoch_records[-1]["avg_task_loss"],
        # Pruning
        "compression_ratio": pruning_stats["compression_ratio"],
        "reduction_percentage": pruning_stats["reduction_percentage"],
        "active_prunable_params": pruning_stats["active_prunable_params"],
        "total_prunable_params": pruning_stats["total_prunable_params"],
        "effective_compression": pruning_stats["effective_compression"],
        **pruning_stats["pruning_percentages"],
        **{f"sparsity_{k}": v for k, v in pruning_stats["sparsity_components"].items()},
    }

    # Cleanup to free GPU memory before next experiment
    del circuit_model, full_model, cached_logits
    torch.cuda.empty_cache()

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


def write_epoch_csv(model_short, task, lambda_sparsity, epoch_records, dirpath):
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, f"{model_short}_{task}_lambda_{lambda_sparsity:.2f}_epochs.csv")
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
    parser = argparse.ArgumentParser(description="Llama lambda_sparsity sweep")
    parser.add_argument(
        "--models", nargs="+",
        default=["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B"],
    )
    parser.add_argument("--tasks", nargs="+", default=["ioi", "gp"], choices=["ioi", "gp", "copycolors"])
    parser.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[0.70, 0.73, 0.75, 0.78, 0.80, 0.83, 0.85, 0.88, 0.90, 0.93, 0.95],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    epoch_dir = os.path.join(args.output_dir, "per_epoch")

    # Read HF token from file if not provided
    hf_token = args.hf_token
    if hf_token is None:
        token_file = os.path.join(PROJECT_ROOT, "hf_tokken.txt")
        if os.path.exists(token_file):
            with open(token_file) as f:
                hf_token = f.read().strip()
            print(f"Loaded HF token from {token_file}")

    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Tasks: {args.tasks}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Flash Attention: {args.flash_attn}")
    print(f"Dry run: {args.dry_run}")

    all_results = []

    for model_name in args.models:
        model_short = model_name.split("/")[-1]
        for task in args.tasks:
            for lam in args.lambdas:
                try:
                    result, epoch_records = run_single_experiment(
                        task=task,
                        model_name=model_name,
                        lambda_sparsity=lam,
                        device=device,
                        hf_token=hf_token,
                        flash_attn=args.flash_attn,
                        dry_run=args.dry_run,
                        seed=args.seed,
                    )
                    all_results.append(result)

                    write_epoch_csv(model_short, task, lam, epoch_records, epoch_dir)

                    # Save after each experiment
                    write_csv(all_results, os.path.join(args.output_dir, "llama_sweep_results.csv"))
                    with open(os.path.join(args.output_dir, "llama_sweep_results.json"), "w") as f:
                        json.dump(all_results, f, indent=2, default=str)

                    print(f"\n  SUMMARY: {model_short}/{task} | ls={lam:.2f}")
                    print(f"    Accuracy: {result['final_accuracy']:.4f} (baseline: {result['baseline_accuracy']:.4f}, drop: {result['accuracy_drop']:.4f})")
                    print(f"    KL Div: {result['final_kl_div']:.4f}")
                    print(f"    Compression: {result['compression_ratio']:.2f}x ({result['reduction_percentage']:.1f}% pruned)")
                    print(f"    Time: {result['training_time_seconds']:.1f}s")

                except Exception as e:
                    print(f"\n  ERROR: {model_name}/{task}/lambda={lam}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Final summary
    print(f"\n{'='*110}")
    print("LLAMA SWEEP COMPLETE - SUMMARY")
    print(f"{'='*110}")
    print(f"{'Model':<16} {'Task':<6} {'Lambda':<8} {'Accuracy':<10} {'Baseline':<10} {'Drop':<8} {'KL Div':<10} {'Pruned%':<10} {'Compress':<10} {'Time(s)':<10}")
    print("-" * 98)
    for r in all_results:
        print(
            f"{r['model_short']:<16} {r['task']:<6} {r['lambda_sparsity']:<8.2f} "
            f"{r['final_accuracy']:<10.4f} {r['baseline_accuracy']:<10.4f} "
            f"{r['accuracy_drop']:<8.4f} {r['final_kl_div']:<10.4f} "
            f"{r['reduction_percentage']:<10.1f} {r['compression_ratio']:<10.2f} "
            f"{r['training_time_seconds']:<10.1f}"
        )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
