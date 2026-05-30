"""
Hyperparameter Sensitivity Analysis: lambda_sparsity sweep across IOI, GP, GT tasks.

Sweeps lambda_sparsity in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95] for each task,
recording accuracy, pruning percentages, sparsity loss, KL divergence, and more.

Outputs:
  - hyperparameter_sweep/results/sweep_results.csv   (flat CSV for plotting)
  - hyperparameter_sweep/results/sweep_results.json   (full structured results)
  - hyperparameter_sweep/results/per_epoch/            (per-epoch training curves)

Usage:
  python -m hyperparameter_sweep.run_lambda_sweep --tasks ioi gp gt --lambdas 0.70 0.725 0.75 0.775 0.80 0.825 0.85 0.875 0.90 0.925 0.95 0.975 1.0
  python -m hyperparameter_sweep.run_lambda_sweep --tasks ioi --lambdas 0.70 0.80 0.90  # subset
"""

import sys
import os
import argparse
import json
import csv
import time
import copy
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Ensure project root is on path
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
# TASK CONFIGS - mirrors the settings from ioi_fast.py, gp.py, gt.py
# =============================================================================

TASK_CONFIGS = {
    "ioi": {
        "num_epochs": 500,
        "learning_rate": 3e-2,
        "batch_size": 32,
        "max_seq_len": 32,
        "use_scheduler": True,
        "scheduler_eta_min": 1e-4,
        "clip_grad": False,
        "pruning_config_overrides": {
            "init_value": 1.0,
            "sparsity_warmup_steps": 500,
            "depth_penalty_scaling": 0.0,
            "lambda_attention_heads": 1.0,
            "lambda_mlp_hidden": 1.0,
            "lambda_mlp_output": 1.0,
            "lambda_attention_neurons": 1.0,
            "lambda_attention_blocks": 1.0,
            "lambda_mlp_blocks": 1.0,
            "prune_full_layers": False,
            "lambda_full_layers": 0.0,
            "prune_embedding": False,
        },
    },
    "docstring": {
        "num_epochs": 500,
        "learning_rate": 3e-2,
        "batch_size": 32,
        "max_seq_len": 64,
        "use_scheduler": False,
        "clip_grad": True,
        "clip_max_norm": 1.0,
        "pruning_config_overrides": {
            "init_value": 0.5,
            "sparsity_warmup_steps": 500,
            "depth_penalty_scaling": 0.0,
            "lambda_attention_heads": 1.0,
            "lambda_mlp_hidden": 1.0,
            "lambda_mlp_output": 1.0,
            "lambda_attention_neurons": 1.0,
            "lambda_attention_blocks": 1.0,
            "lambda_mlp_blocks": 2.0,
            "prune_full_layers": False,
            "lambda_full_layers": 0.0,
            "prune_embedding": False,
        },
    },
    "gp": {
        "num_epochs": 500,
        "learning_rate": 3e-1,
        "batch_size": 64,
        "max_seq_len": 32,
        "use_scheduler": False,
        "clip_grad": True,
        "clip_max_norm": 1.0,
        "pruning_config_overrides": {
            "init_value": 0.5,
            "sparsity_warmup_steps": 1000,
            "depth_penalty_scaling": 0.0,
            "lambda_attention_heads": 1.0,
            "lambda_mlp_hidden": 1.0,
            "lambda_mlp_output": 1.0,
            "lambda_attention_neurons": 1.0,
            "lambda_attention_blocks": 1.0,
            "lambda_mlp_blocks": 2.0,
            "prune_full_layers": False,
            "lambda_full_layers": 0.0,
            "prune_embedding": False,
        },
    },
    "gt": {
        "num_epochs": 250,
        "learning_rate": 5e-2,
        "batch_size": 16,
        "max_seq_len": 32,
        "use_scheduler": False,
        "clip_grad": True,
        "clip_max_norm": 1.0,
        "pruning_config_overrides": {
            "init_value": 0.5,
            "sparsity_warmup_steps": 1000,
            "depth_penalty_scaling": 0.0,
            "lambda_attention_heads": 0.0,
            "lambda_mlp_hidden": 1.0,
            "lambda_mlp_output": 1.0,
            "lambda_attention_neurons": 1.0,
            "lambda_attention_blocks": 1.0,
            "lambda_mlp_blocks": 1.0,
            "prune_full_layers": False,
            "lambda_full_layers": 0.0,
            "prune_embedding": False,
        },
    },
}


# =============================================================================
# DATA LOADING (task-specific, mirrors the original scripts exactly)
# =============================================================================

def load_task_data(task: str, tokenizer, full_model, device: str, cfg: dict):
    """Load and filter data for a task. Returns (train_loader, val_loader, test_loader, extras)."""
    batch_size = cfg["batch_size"]
    max_seq_len = cfg["max_seq_len"]

    if task == "ioi":
        from dataset.ioi import (
            IOIDataset,
            load_or_generate_ioi_data,
            run_evaluation,
            filter_dataset_by_model_correctness,
        )

        train_data = load_or_generate_ioi_data(split="train", num_samples=200)
        val_data = load_or_generate_ioi_data(split="validation", num_samples=200)
        test_data = load_or_generate_ioi_data(split="test", num_samples=1000)

        val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device, batch_size=batch_size)
        test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, device, batch_size=batch_size)

        train_loader = DataLoader(IOIDataset(train_data, tokenizer), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(IOIDataset(val_data, tokenizer), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(IOIDataset(test_data, tokenizer), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, {"eval_fn": run_evaluation}

    elif task == "gp":
        from dataset.gp import (
            GPDataset,
            load_or_generate_gp_data,
            run_evaluation,
            filter_dataset_by_model_correctness,
        )

        train_data = load_or_generate_gp_data(split="train", num_samples=100000)
        val_data = load_or_generate_gp_data(split="validation", num_samples=10000)
        test_data = load_or_generate_gp_data(split="test", num_samples=100000)

        val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device, max_length=max_seq_len, batch_size=batch_size)
        test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, device, max_length=max_seq_len, batch_size=batch_size)

        train_loader = DataLoader(GPDataset(train_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GPDataset(val_data, tokenizer, max_length=max_seq_len), batch_size=batch_size)
        test_loader = DataLoader(GPDataset(test_data, tokenizer, max_length=max_seq_len), batch_size=batch_size)

        return train_loader, val_loader, test_loader, {"eval_fn": run_evaluation}

    elif task == "gt":
        from dataset.gt_gpt2 import (
            GTDataset,
            load_or_generate_gt_data,
            create_two_digit_token_mapping,
            run_evaluation,
            filter_dataset_by_model_correctness,
        )

        two_digit_tokens = create_two_digit_token_mapping(tokenizer)

        train_data = load_or_generate_gt_data(split="train", num_samples=200)
        val_data = load_or_generate_gt_data(split="validation", num_samples=200)
        test_data = load_or_generate_gt_data(split="test", num_samples=1000)

        # Normalize keys: GTDataset expects 'prefix' but generated data only has 'clean_prompt'
        for data_list in [train_data, val_data, test_data]:
            for sample in data_list:
                if "prefix" not in sample and "clean_prompt" in sample:
                    sample["prefix"] = sample["clean_prompt"]

        train_data = filter_dataset_by_model_correctness(train_data, full_model, tokenizer, device, two_digit_tokens)
        val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device, two_digit_tokens)
        test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, device, two_digit_tokens)

        train_loader = DataLoader(GTDataset(train_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GTDataset(val_data, tokenizer, max_length=max_seq_len), batch_size=batch_size)
        test_loader = DataLoader(GTDataset(test_data, tokenizer, max_length=max_seq_len), batch_size=batch_size)

        return train_loader, val_loader, test_loader, {"eval_fn": run_evaluation, "two_digit_tokens": two_digit_tokens}

    elif task == "docstring":
        from dataset.docstring import (
            DocstringDataset,
            generate_docstring_data,
            run_evaluation,
            filter_dataset_by_model_correctness,
        )

        train_data = generate_docstring_data(num_samples=5000, seed=42)
        val_data = generate_docstring_data(num_samples=500, seed=123)
        test_data = generate_docstring_data(num_samples=1000, seed=456)

        val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device, max_length=max_seq_len, batch_size=batch_size)
        test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, device, max_length=max_seq_len, batch_size=batch_size)

        train_loader = DataLoader(DocstringDataset(train_data, tokenizer, max_length=max_seq_len), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(DocstringDataset(val_data, tokenizer, max_length=max_seq_len), batch_size=batch_size)
        test_loader = DataLoader(DocstringDataset(test_data, tokenizer, max_length=max_seq_len), batch_size=batch_size)

        return train_loader, val_loader, test_loader, {"eval_fn": run_evaluation}

    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# TASK-SPECIFIC TRAINING STEPS (mirrors original scripts exactly)
# =============================================================================

def train_step_ioi(circuit_model, full_model, batch, device):
    """Single training step for IOI task. Returns (kl_loss, task_loss)."""
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
        t_start = batch["T_Start"][i].item() - 1
        t_end = batch["T_End"][i].item() - 1
        valid_length = batch["attention_mask"][i].sum().item()
        end_pos = min(t_end, valid_length)

        if t_start < end_pos:
            circuit_logits = circuit_outputs.logits[i, t_start]
            target_logits = target_outputs.logits[i, t_start]
            kl = F.kl_div(
                F.log_softmax(circuit_logits, dim=-1),
                F.log_softmax(target_logits, dim=-1),
                reduction="sum",
                log_target=True,
            )
            total_kl += kl

    # Task loss
    pos_good = batch["T_Start"] - 1
    pos_bad = batch["D_Start"] - 1
    token_good = batch["target_tokens"][:, 0]
    token_bad = batch["distractor_tokens"][:, 0]
    batch_indices = torch.arange(batch_size_curr, device=device)

    logit_good = circuit_outputs.logits[batch_indices, pos_good, token_good]
    logit_bad = circuit_outputs.logits[batch_indices, pos_bad, token_bad]
    task_loss = F.relu(1.0 - (logit_good - logit_bad)).mean()

    kl_loss = total_kl / batch_size_curr
    return kl_loss, task_loss


def train_step_gp(circuit_model, full_model, batch, device):
    """Single training step for GP task. Returns (kl_loss, task_loss)."""
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

    batch_size = circuit_outputs.logits.size(0)
    total_kl = 0

    for i in range(batch_size):
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
        torch.arange(batch_size), batch["prefix_length"] - 1, batch["target_token"]
    ]
    logit_bad = circuit_outputs.logits[
        torch.arange(batch_size), batch["prefix_length"] - 1, batch["distractor_token"]
    ]
    task_loss = F.relu(0.1 - (logit_good - logit_bad)).mean()

    kl_loss = total_kl / batch_size
    return kl_loss, task_loss


def train_step_gt(circuit_model, full_model, batch, device, digit_token_ids):
    """Single training step for GT task. Returns (kl_loss, task_loss=0)."""
    circuit_outputs = circuit_model(
        input_ids=batch["clean_input_ids"],
        corrupted_input_ids=batch["corrupted_input_ids"],
        attention_mask=batch["clean_attention_mask"],
    )

    with torch.no_grad():
        target_outputs = full_model(
            input_ids=batch["clean_input_ids"],
            attention_mask=batch["clean_attention_mask"],
        )

    last_token_circuit_logits = circuit_outputs.logits[
        torch.arange(circuit_outputs.logits.size(0)), batch["last_token_idx"], :
    ]
    last_token_target_logits = target_outputs.logits[
        torch.arange(target_outputs.logits.size(0)), batch["last_token_idx"], :
    ]

    digit_logits_circuit = torch.gather(
        last_token_circuit_logits,
        1,
        digit_token_ids.unsqueeze(0).expand(last_token_circuit_logits.shape[0], -1),
    )
    digit_logits_target = torch.gather(
        last_token_target_logits,
        1,
        digit_token_ids.unsqueeze(0).expand(last_token_target_logits.shape[0], -1),
    )

    kl_loss = F.kl_div(
        F.log_softmax(digit_logits_circuit, dim=-1),
        F.log_softmax(digit_logits_target, dim=-1),
        reduction="batchmean",
        log_target=True,
    )

    # GT has no separate task loss
    task_loss = torch.tensor(0.0, device=device)
    return kl_loss, task_loss


def train_step_docstring(circuit_model, full_model, batch, device):
    """Single training step for Docstring task. Returns (kl_loss, task_loss)."""
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

    batch_size = circuit_outputs.logits.size(0)
    total_kl = 0

    for i in range(batch_size):
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
        torch.arange(batch_size), batch["prefix_length"] - 1, batch["target_token"]
    ]
    logit_bad = circuit_outputs.logits[
        torch.arange(batch_size), batch["prefix_length"] - 1, batch["distractor_token"]
    ]
    task_loss = F.relu(0.1 - (logit_good - logit_bad)).mean()

    kl_loss = total_kl / batch_size
    return kl_loss, task_loss


# =============================================================================
# EVALUATION WRAPPERS
# =============================================================================

def run_task_evaluation(task, model, full_model, dataloader, device, tokenizer, extras, name=""):
    """Run evaluation for a task, return unified results dict."""
    eval_fn = extras["eval_fn"]

    if task == "gt":
        results = eval_fn(
            model_to_eval=model,
            model_name=name,
            full_model_for_faithfulness=full_model,
            dataloader=dataloader,
            device=device,
            two_digit_tokens=extras["two_digit_tokens"],
            tokenizer=tokenizer,
        )
        return {
            "accuracy": results.get("accuracy", 0.0),
            "logit_diff": results.get("cutoff_sharpness", 0.0),
            "kl_div": results.get("kl_div", 0.0),
            "exact_match": 0.0,
            "metric_name": "prob_diff",  # so we know what "accuracy" actually is
        }
    else:
        results = eval_fn(
            model_to_eval=model,
            model_name=name,
            full_model_for_faithfulness=full_model,
            dataloader=dataloader,
            device=device,
            tokenizer=tokenizer,
        )
        return {
            "accuracy": results.get("accuracy", 0.0),
            "logit_diff": results.get("logit_diff", 0.0),
            "kl_div": results.get("kl_div", 0.0),
            "exact_match": results.get("exact_match", 0.0),
            "metric_name": "accuracy",
        }


# =============================================================================
# PRUNING STATISTICS EXTRACTION
# =============================================================================

def extract_pruning_stats(circuit_model, total_steps):
    """Extract detailed pruning statistics from a trained circuit model."""
    circuit_model.eval()

    # Get sparsity loss components (before finalization)
    sparsity_info = circuit_model.get_sparsity_loss(step=total_steps)
    sparsity_components = {k: v.item() if torch.is_tensor(v) else v for k, v in sparsity_info.items()}

    # Finalize and get hierarchical pruning report
    finalize_results = analyze_and_finalize_circuit(circuit_model, verbose=False)

    granularity = finalize_results["granularity_stats"]
    prunable = finalize_results["prunable_compression"]

    # Compute pruning percentages per granularity
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
# MAIN TRAINING + EVALUATION LOOP FOR ONE (task, lambda) PAIR
# =============================================================================

def run_single_experiment(task: str, lambda_sparsity: float, device: str, seed: int = 42):
    """
    Train a circuit model for one task with a specific lambda_sparsity.
    Returns a dict with all metrics for this run.
    """
    torch.manual_seed(seed)

    cfg = TASK_CONFIGS[task]
    model_name = "gpt2"

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT: task={task}, lambda_sparsity={lambda_sparsity:.2f}")
    print(f"{'='*80}")

    # --- Setup tokenizer and models ---
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build pruning config from task defaults
    pruning_kwargs = {k: v for k, v in cfg["pruning_config_overrides"].items()}
    pruning_config = PruningConfig(**pruning_kwargs)

    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(model_name, pruning_config).to(device).eval()
    full_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    disable_dropout(circuit_model)

    # Freeze base weights, unfreeze gates
    for name, param in circuit_model.named_parameters():
        param.requires_grad = "gate" in name

    # --- Load data ---
    train_loader, val_loader, test_loader, extras = load_task_data(
        task, tokenizer, full_model, device, cfg
    )

    # --- Baseline evaluation ---
    print("Running baseline evaluation...")
    baseline_results = run_task_evaluation(
        task, full_model, None, test_loader, device, tokenizer, extras, name="Baseline"
    )
    print(f"  Baseline: accuracy={baseline_results['accuracy']:.4f}, logit_diff={baseline_results['logit_diff']:.4f}")

    # --- Pre-compute GT digit tokens if needed ---
    digit_token_ids = None
    if task == "gt":
        from dataset.gt_gpt2 import create_two_digit_token_mapping
        two_digit_tokens = extras["two_digit_tokens"]
        sorted_tokens = sorted(two_digit_tokens.items())
        digit_token_ids = torch.tensor([item[1] for item in sorted_tokens], device=device)

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=cfg["learning_rate"])

    scheduler = None
    if cfg.get("use_scheduler"):
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=cfg.get("scheduler_eta_min", 1e-4))

    circuit_model.train()
    total_steps = 0
    epoch_records = []  # per-epoch training curves

    train_step_fn = {
        "ioi": lambda batch: train_step_ioi(circuit_model, full_model, batch, device),
        "gp": lambda batch: train_step_gp(circuit_model, full_model, batch, device),
        "gt": lambda batch: train_step_gt(circuit_model, full_model, batch, device, digit_token_ids),
        "docstring": lambda batch: train_step_docstring(circuit_model, full_model, batch, device),
    }[task]

    start_time = time.time()
    epoch_pbar = tqdm(range(cfg["num_epochs"]), desc=f"{task} ls={lambda_sparsity:.2f}")

    for epoch in epoch_pbar:
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_sparsity = 0.0
        epoch_task_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move batch to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            kl_loss, task_loss = train_step_fn(batch)
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)["total_sparsity"]

            # Combined loss (matches original scripts)
            if task == "gt":
                loss = (1 - lambda_sparsity) * kl_loss + lambda_sparsity * sparsity_loss
            else:
                loss = (1 - lambda_sparsity) * (kl_loss + task_loss) + lambda_sparsity * sparsity_loss

            loss.backward()

            if cfg.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(gate_params, max_norm=cfg.get("clip_max_norm", 1.0))

            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_sparsity += sparsity_loss.item()
            epoch_task_loss += task_loss.item()
            total_steps += 1

        if scheduler is not None:
            scheduler.step()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_kl = epoch_kl / n_batches
        avg_sparsity = epoch_sparsity / n_batches
        avg_task_loss = epoch_task_loss / n_batches

        epoch_pbar.set_postfix({
            "L": f"{avg_loss:.3f}",
            "KL": f"{avg_kl:.3f}",
            "Sp": f"{avg_sparsity:.3f}",
        })

        # Record every epoch for training curves
        epoch_record = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_kl_loss": avg_kl,
            "avg_sparsity_loss": avg_sparsity,
            "avg_task_loss": avg_task_loss,
        }

        # Validation every 50 epochs (lighter than every 10 to save time in sweep)
        if (epoch + 1) % 50 == 0:
            circuit_model.eval()
            val_results = run_task_evaluation(
                task, circuit_model, full_model, val_loader, device, tokenizer, extras,
                name=f"Val Ep {epoch+1}",
            )
            epoch_record["val_accuracy"] = val_results["accuracy"]
            epoch_record["val_kl_div"] = val_results["kl_div"]
            epoch_record["val_logit_diff"] = val_results["logit_diff"]
            circuit_model.train()

        epoch_records.append(epoch_record)

    training_time = time.time() - start_time

    # --- Final evaluation on TEST set ---
    print("\nRunning final evaluation...")
    circuit_model.eval()

    # Get pruning stats (this also calls analyze_and_finalize_circuit)
    pruning_stats = extract_pruning_stats(circuit_model, total_steps)

    # Final test evaluation (model is now in final circuit mode)
    final_results = run_task_evaluation(
        task, circuit_model, full_model, test_loader, device, tokenizer, extras,
        name=f"Final ls={lambda_sparsity:.2f}",
    )

    # --- Compile results ---
    result = {
        "task": task,
        "lambda_sparsity": lambda_sparsity,
        "seed": seed,
        "num_epochs": cfg["num_epochs"],
        "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "training_time_seconds": training_time,
        # Baseline metrics
        "baseline_accuracy": baseline_results["accuracy"],
        "baseline_logit_diff": baseline_results["logit_diff"],
        # Final test metrics
        "final_accuracy": final_results["accuracy"],
        "final_logit_diff": final_results["logit_diff"],
        "final_kl_div": final_results["kl_div"],
        "final_exact_match": final_results["exact_match"],
        "accuracy_drop": baseline_results["accuracy"] - final_results["accuracy"],
        "metric_name": final_results["metric_name"],
        # Training loss at final epoch
        "final_train_loss": epoch_records[-1]["avg_loss"],
        "final_train_kl": epoch_records[-1]["avg_kl_loss"],
        "final_train_sparsity": epoch_records[-1]["avg_sparsity_loss"],
        "final_train_task_loss": epoch_records[-1]["avg_task_loss"],
        # Pruning statistics
        "compression_ratio": pruning_stats["compression_ratio"],
        "reduction_percentage": pruning_stats["reduction_percentage"],
        "active_prunable_params": pruning_stats["active_prunable_params"],
        "total_prunable_params": pruning_stats["total_prunable_params"],
        "effective_compression": pruning_stats["effective_compression"],
        # Per-granularity pruning percentages
        **pruning_stats["pruning_percentages"],
        # Sparsity loss components
        **{f"sparsity_{k}": v for k, v in pruning_stats["sparsity_components"].items()},
    }

    return result, epoch_records


# =============================================================================
# CSV WRITING
# =============================================================================

def write_csv(results: List[Dict], filepath: str):
    """Write list of flat dicts to CSV."""
    if not results:
        return
    # Collect all keys across all results
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


def write_epoch_csv(task: str, lambda_sparsity: float, epoch_records: List[Dict], dirpath: str):
    """Write per-epoch training curves to CSV."""
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, f"{task}_lambda_{lambda_sparsity:.2f}_epochs.csv")
    if not epoch_records:
        return
    keys = list(epoch_records[0].keys())
    # Union of all keys (later epochs may have val metrics)
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
    parser = argparse.ArgumentParser(description="Lambda sparsity hyperparameter sweep")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["ioi", "gp", "gt", "docstring"],
        choices=["ioi", "gp", "gt", "docstring"],
        help="Tasks to sweep over",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        help="Lambda sparsity values to sweep",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    epoch_dir = os.path.join(args.output_dir, "per_epoch")
    os.makedirs(epoch_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Tasks: {args.tasks}")
    print(f"Lambda values: {args.lambdas}")
    print(f"Output directory: {args.output_dir}")

    all_results = []

    for task in args.tasks:
        for lam in args.lambdas:
            try:
                result, epoch_records = run_single_experiment(
                    task=task,
                    lambda_sparsity=lam,
                    device=device,
                    seed=args.seed,
                )
                all_results.append(result)

                # Save epoch curves
                write_epoch_csv(task, lam, epoch_records, epoch_dir)

                # Save running results after each experiment (in case of crash)
                write_csv(all_results, os.path.join(args.output_dir, "sweep_results.csv"))
                with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

                # Print summary for this run
                print(f"\n  SUMMARY: {task} | ls={lam:.2f}")
                print(f"    Accuracy: {result['final_accuracy']:.4f} (baseline: {result['baseline_accuracy']:.4f}, drop: {result['accuracy_drop']:.4f})")
                print(f"    KL Div: {result['final_kl_div']:.4f}")
                print(f"    Compression: {result['compression_ratio']:.2f}x ({result['reduction_percentage']:.1f}% pruned)")
                print(f"    Training time: {result['training_time_seconds']:.1f}s")

            except Exception as e:
                print(f"\n  ERROR: task={task}, lambda={lam}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Final summary table
    print(f"\n{'='*100}")
    print("SWEEP COMPLETE - SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Task':<6} {'Lambda':<8} {'Accuracy':<10} {'Baseline':<10} {'Drop':<8} {'KL Div':<10} {'Pruned%':<10} {'Compress':<10} {'Time(s)':<10}")
    print("-" * 82)
    for r in all_results:
        print(
            f"{r['task']:<6} {r['lambda_sparsity']:<8.2f} "
            f"{r['final_accuracy']:<10.4f} {r['baseline_accuracy']:<10.4f} "
            f"{r['accuracy_drop']:<8.4f} {r['final_kl_div']:<10.4f} "
            f"{r['reduction_percentage']:<10.1f} {r['compression_ratio']:<10.2f} "
            f"{r['training_time_seconds']:<10.1f}"
        )

    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - sweep_results.csv  (for plotting)")
    print(f"  - sweep_results.json (full structured data)")
    print(f"  - per_epoch/         (training curves per experiment)")


if __name__ == "__main__":
    main()
