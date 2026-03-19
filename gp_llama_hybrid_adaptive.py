"""
Gender Pronouns Circuit Discovery with Hybrid Adaptive Pruning for Llama.

Adapted from IOI Llama implementation for the Gender Pronouns task.

Usage:
    # Auto-discover sparsity, maintain 95% baseline accuracy
    python gp_llama_hybrid_adaptive.py --target-accuracy 0.95

    # Fully automatic (both accuracy and sparsity adaptive)
    python gp_llama_hybrid_adaptive.py --fully-adaptive

    # With speedups
    python gp_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import time
import argparse
import os
import numpy as np
from collections import deque

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from dataset.gp_llama import (
    GPDatasetLlama,
    load_or_generate_gp_data,
    run_evaluation,
    filter_dataset_by_model_correctness,
)
from utils import disable_dropout, analyze_and_finalize_circuit
from dataclasses import dataclass

# Import the HybridAdaptiveScheduler from IOI
from ioi_llama_hybrid_adaptive import HybridAdaptiveConfig, HybridAdaptiveScheduler, HybridLlamaPruningConfig


# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_overall_sparsity(model) -> float:
    """Compute overall sparsity rate."""
    from models.l0 import HardConcreteGate

    total_gates = 0
    open_gates = 0

    for module in model.modules():
        if isinstance(module, HardConcreteGate):
            with torch.no_grad():
                gates = module()
                total_gates += gates.numel()
                open_gates += (gates > 0.5).sum().item()

    return 1.0 - (open_gates / total_gates) if total_gates > 0 else 0.0


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid Adaptive Circuit Discovery for Gender Pronouns")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints_llama_gp_hybrid')

    # Key parameter: target accuracy (None = fully adaptive)
    parser.add_argument('--target-accuracy', type=float, default=0.95,
                        help='Target accuracy as fraction of baseline (e.g., 0.95 = 95%%). Use --fully-adaptive to disable.')
    parser.add_argument('--fully-adaptive', action='store_true',
                        help='Fully adaptive mode (no target accuracy)')

    # Speedups
    parser.add_argument('--flash-attn', action='store_true')

    args = parser.parse_args()

    # Read HF token
    hf_token = args.hf_token
    if hf_token is None:
        token_file = "hf_tokken.txt"
        if os.path.exists(token_file):
            with open(token_file) as f:
                hf_token = f.read().strip()

    # Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("  HYBRID ADAPTIVE CIRCUIT DISCOVERY - GENDER PRONOUNS")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Flash Attention: {args.flash_attn}")

    # Load models
    print("\n--- Loading models ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    pruning_config = HybridLlamaPruningConfig()
    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model, pruning_config, **model_kwargs
    ).to(DEVICE).eval()

    full_model = LlamaForCausalLM.from_pretrained(
        args.model, **model_kwargs
    ).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    disable_dropout(circuit_model)

    # Freeze base, unfreeze gates
    GATE_PATTERNS = ('_gates.', '_gate.', 'embedding_gate.', 'layer_gates.')
    for name, param in circuit_model.named_parameters():
        is_gate = any(p in name for p in GATE_PATTERNS)
        param.requires_grad = is_gate
        if is_gate:
            param.data = param.data.float()

    # Data
    NUM_TRAIN = 10 if args.dry_run else 200
    NUM_VAL = 5 if args.dry_run else 200
    NUM_TEST = 5 if args.dry_run else 1000

    train_data = load_or_generate_gp_data(split="train", num_samples=NUM_TRAIN)
    val_data = load_or_generate_gp_data(split="validation", num_samples=NUM_VAL)
    test_data = load_or_generate_gp_data(split="test", num_samples=NUM_TEST)

    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, DEVICE, args.batch_size)
    test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, DEVICE, args.batch_size)

    train_dataset = GPDatasetLlama(train_data, tokenizer)
    val_dataset = GPDatasetLlama(val_data, tokenizer)
    test_dataset = GPDatasetLlama(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Baseline
    print("\n--- Baseline evaluation ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline",
        full_model_for_faithfulness=None,
        dataloader=test_dataloader,
        device=DEVICE,
        verbose=True,
        tokenizer=tokenizer,
    )
    base_accuracy = baseline_results['accuracy']
    print(f"🎯 Baseline: {base_accuracy:.4f}")

    # Initialize scheduler
    scheduler_config = HybridAdaptiveConfig(
        warmup_steps=pruning_config.sparsity_warmup_steps,
        target_accuracy=None if args.fully_adaptive else args.target_accuracy,
    )
    scheduler = HybridAdaptiveScheduler(scheduler_config, base_accuracy)

    # Setup training
    optimizer = AdamW([p for p in circuit_model.parameters() if p.requires_grad], lr=args.lr)

    # Pre-cache full model
    print("\n🚀 Caching full model outputs...")
    cached_train_logits = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = full_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False)
            cached_train_logits[batch_idx] = out.logits.detach()

    # Training loop
    print(f"\n{'='*80}")
    print("  TRAINING")
    print(f"{'='*80}\n")

    total_steps = 0
    NUM_EPOCHS = 2 if args.dry_run else args.epochs

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        circuit_model.train()

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )

            # KL loss - computed at the prediction position
            total_kl = 0
            batch_size = outputs.logits.size(0)
            for i in range(batch_size):
                pred_pos = batch['prefix_length'][i].item() - 1

                # Get valid sequence length
                valid_length = batch['attention_mask'][i].sum().item()

                # Only compute KL if within valid range
                if pred_pos < valid_length:
                    kl = F.kl_div(
                        F.log_softmax(outputs.logits[i, pred_pos].float(), dim=-1),
                        F.log_softmax(cached_train_logits[batch_idx][i, pred_pos].float(), dim=-1),
                        reduction='sum', log_target=True,
                    )
                    total_kl += kl
            kl_loss = total_kl / batch_size

            # Task loss - logit difference
            logit_target = outputs.logits[
                torch.arange(batch_size, device=DEVICE),
                batch['prefix_length'] - 1,
                batch['target_token']
            ].float()

            logit_distractor = outputs.logits[
                torch.arange(batch_size, device=DEVICE),
                batch['prefix_length'] - 1,
                batch['distractor_token']
            ].float()

            task_loss = F.relu(4.0 - (logit_target - logit_distractor)).mean()

            # Sparsity loss with adaptive multiplier
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            sparsity_loss = sparsity_loss * scheduler.lambda_multiplier

            loss = kl_loss * 1.0 + sparsity_loss # + task_loss
            loss.backward()
            optimizer.step()

            total_steps += 1

        # Validation
        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Ep{epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_dataloader,
                device=DEVICE,
                verbose=False,
                tokenizer=tokenizer,
            )

            current_sparsity = compute_overall_sparsity(circuit_model)

            scheduler.step_update(
                step=total_steps,
                epoch=epoch + 1,
                accuracy=val_results['accuracy'],
                sparsity_rate=current_sparsity,
                kl_loss=val_results['kl_div'],
            )

            if scheduler.should_stop_early():
                print("\n🎉 Converged! Stopping early.")
                break

    # Save plot
    scheduler.plot_training_dynamics(os.path.join(args.save_dir, 'training.png'))

    # Final eval
    print("\n--- Final Evaluation ---")
    circuit_model.eval()
    analyze_and_finalize_circuit(circuit_model)
    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Final",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        verbose=True,
        tokenizer=tokenizer,
    )

    summary = scheduler.get_final_summary()
    print(f"\n{'='*80}")
    print("  SUMMARY - GENDER PRONOUNS")
    print(f"{'='*80}")
    print(f"Mode: {summary['mode']}")
    print(f"Baseline: {summary['baseline_accuracy']:.4f}")
    if summary['target_accuracy']:
        print(f"Target:   {summary['target_accuracy']:.4f}")
    print(f"Final:    {summary['final_accuracy']:.4f} ({summary['accuracy_drop_pct']:.1f}% drop)")
    print(f"Sparsity: {summary['final_sparsity']:.4f}")
    print(f"Epochs:   {summary['total_epochs']}")
    print(f"{'='*80}\n")
