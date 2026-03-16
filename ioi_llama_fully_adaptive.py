"""
IOI Circuit Discovery for Llama with FULLY Adaptive Pruning.

Zero manual configuration! Just run and the scheduler automatically:
- Finds optimal accuracy/sparsity tradeoff
- Adjusts pruning pressure based on training dynamics
- Stops when converged

Usage:
    python ioi_llama_fully_adaptive.py                    # Just run it!
    python ioi_llama_fully_adaptive.py --dry-run          # Quick test
    python ioi_llama_fully_adaptive.py --flash-attn       # Fast mode
    python ioi_llama_fully_adaptive.py --conservative     # Keep higher accuracy
    python ioi_llama_fully_adaptive.py --aggressive       # Maximize sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import random
import time
import argparse
import os

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from dataset.ioi_llama import (
    IOIDatasetLlama,
    generate_ioi_data_llama,
    run_evaluation,
    filter_dataset_by_model_correctness,
)
from utils import disable_dropout, analyze_and_finalize_circuit
from pruning_scheduler_v2 import FullyAdaptivePruningScheduler, FullyAdaptiveConfig
from dataclasses import dataclass


# ==============================================================================
# FULLY ADAPTIVE PRUNING CONFIGURATION
# ==============================================================================

@dataclass
class FullyAdaptiveLlamaPruningConfig(PruningConfig):
    """
    PruningConfig for fully adaptive training.

    Base lambdas are just starting points - scheduler will adapt them.
    Set these to 1.0 and let the multiplier do the work.
    """
    init_value: float = 0.5
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.0

    # All base lambdas = 1.0 (scheduler controls via multiplier)
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


def compute_overall_sparsity(model) -> float:
    """Compute overall sparsity rate across all gates."""
    from models.l0 import HardConcreteGate

    total_gates = 0
    open_gates = 0

    for module in model.modules():
        if isinstance(module, HardConcreteGate):
            with torch.no_grad():
                gates = module()
                total_gates += gates.numel()
                open_gates += (gates > 0.5).sum().item()

    sparsity_rate = 1.0 - (open_gates / total_gates) if total_gates > 0 else 0.0
    return sparsity_rate


def _save_checkpoint(circuit_model, optimizer, epoch, total_steps, scheduler, path, gate_patterns):
    """Save checkpoint including scheduler state."""
    gate_state = {}
    for name, param in circuit_model.named_parameters():
        if any(p in name for p in gate_patterns):
            gate_state[name] = param.data.cpu()

    checkpoint = {
        'epoch': epoch,
        'total_steps': total_steps,
        'gate_state_dict': gate_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state': {
            'lambda_multiplier': scheduler.lambda_multiplier,
            'best_lambda': scheduler.best_lambda,
            'best_sparsity': scheduler.best_sparsity_at_acceptable_acc,
            'phase': scheduler.phase,
        },
    }
    torch.save(checkpoint, path)
    print(f"  💾 Checkpoint saved: {path}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fully Adaptive IOI Circuit Discovery")
    parser.add_argument('--dry-run', action='store_true', help='Quick test')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs (early stop likely)')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints_llama_fully_adaptive')
    parser.add_argument('--save-every', type=int, default=50)

    # Adaptation style
    parser.add_argument('--conservative', action='store_true',
                        help='Keep higher accuracy (min 90% of baseline)')
    parser.add_argument('--aggressive', action='store_true',
                        help='Push for maximum sparsity (min 80% of baseline)')

    # Speedups
    parser.add_argument('--flash-attn', action='store_true',
                        help='Use Flash Attention 2')

    args = parser.parse_args()

    # --- Read HF token ---
    hf_token = args.hf_token
    if hf_token is None:
        token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_tokken.txt")
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                hf_token = f.read().strip()

    # --- Configuration ---
    MODEL_NAME = args.model
    NUM_EPOCHS = 2 if args.dry_run else args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    NUM_TRAIN = 10 if args.dry_run else 200
    NUM_VAL = 5 if args.dry_run else 200
    NUM_TEST = 5 if args.dry_run else 1000

    print("="*80)
    print("  FULLY ADAPTIVE CIRCUIT DISCOVERY")
    print("  No manual targets needed - automatic optimization!")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Flash Attention: {args.flash_attn}")

    # Set adaptation style
    if args.conservative:
        min_acc_fraction = 0.90
        print(f"Mode: CONSERVATIVE (min accuracy: 90% of baseline)")
    elif args.aggressive:
        min_acc_fraction = 0.80
        print(f"Mode: AGGRESSIVE (min accuracy: 80% of baseline)")
    else:
        min_acc_fraction = 0.85
        print(f"Mode: BALANCED (min accuracy: 85% of baseline)")

    pruning_config = FullyAdaptiveLlamaPruningConfig()

    # --- Model and Tokenizer Setup ---
    print("\n--- Loading tokenizer and models ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  ⚡ Using Flash Attention 2")

    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL_NAME, pruning_config, **model_kwargs
    ).to(DEVICE).eval()

    full_model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, **model_kwargs
    ).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    disable_dropout(circuit_model)

    # --- Freeze base model, unfreeze gates ---
    print("Freezing base model weights and unfreezing gate parameters...")
    total_params = 0
    trainable_params = 0
    GATE_PATTERNS = ('_gates.', '_gate.', 'embedding_gate.', 'layer_gates.')

    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        is_pruning_gate = any(p in name for p in GATE_PATTERNS)
        if not is_pruning_gate:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()
            trainable_params += param.numel()

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable gates: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\n--- Setting up IOI dataset ---")
    train_data = generate_ioi_data_llama(NUM_TRAIN, tokenizer, seed=42)
    val_data = generate_ioi_data_llama(NUM_VAL, tokenizer, seed=123)
    test_data = generate_ioi_data_llama(NUM_TEST, tokenizer, seed=456)

    print("\n--- Filtering datasets ---")
    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, DEVICE, BATCH_SIZE)
    test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, DEVICE, BATCH_SIZE)

    train_dataset = IOIDatasetLlama(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = IOIDatasetLlama(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = IOIDatasetLlama(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    # --- Baseline Evaluation ---
    print("\n--- Baseline evaluation ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline",
        full_model_for_faithfulness=None,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )
    base_accuracy = baseline_results.get("accuracy", 0.0)
    print(f"\n🎯 Baseline Accuracy: {base_accuracy:.4f}")

    # --- Initialize Fully Adaptive Scheduler ---
    print("\n--- Initializing Fully Adaptive Scheduler ---")
    scheduler_config = FullyAdaptiveConfig(
        warmup_steps=pruning_config.sparsity_warmup_steps,
        min_accuracy_fraction=min_acc_fraction,
    )
    scheduler = FullyAdaptivePruningScheduler(scheduler_config, base_accuracy)

    print(f"  Scheduler will automatically find optimal sparsity!")
    print(f"  Minimum acceptable accuracy: {scheduler.min_acceptable_accuracy:.4f}")
    print(f"  Will stop early when converged (~{scheduler_config.min_training_epochs}-300 epochs typical)")

    # --- Training Setup ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)

    start_epoch = 0
    total_steps = 0

    # --- Pre-cache full model outputs ---
    print("\n🚀 Pre-caching full model outputs...")
    cached_train_logits = {}
    full_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Caching")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)
            out = full_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )
            cached_train_logits[batch_idx] = out.logits.detach()
    print(f"  Cached {len(cached_train_logits)} batches")

    circuit_model.train()

    # --- Training Loop ---
    print(f"\n{'='*80}")
    print(f"  FULLY ADAPTIVE TRAINING - LET'S DISCOVER THE CIRCUIT!")
    print(f"{'='*80}\n")

    epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS), desc="Training", initial=start_epoch, total=NUM_EPOCHS)

    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)

            # Forward pass
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )

            target_logits = cached_train_logits[batch_idx]

            # KL loss
            batch_size_curr = circuit_outputs.logits.size(0)
            total_kl = 0

            for i in range(batch_size_curr):
                t_start = batch['T_Start'][i].item() - 1
                t_end = batch['T_End'][i].item() - 1
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, int(valid_length))

                if t_start < end_pos:
                    circuit_logits_slice = circuit_outputs.logits[i, t_start:end_pos].float()
                    target_logits_slice = target_logits[i, t_start:end_pos].float()

                    kl = F.kl_div(
                        F.log_softmax(circuit_logits_slice, dim=-1),
                        F.log_softmax(target_logits_slice, dim=-1),
                        reduction='sum',
                        log_target=True,
                    )
                    total_kl += kl

            # Task loss
            pos_good = batch['T_Start'] - 1
            pos_bad = batch['D_Start'] - 1
            token_good = batch['target_tokens'][:, 0]
            token_bad = batch['distractor_tokens'][:, 0]
            batch_indices = torch.arange(batch_size_curr, device=DEVICE)

            logit_good = circuit_outputs.logits[batch_indices, pos_good, token_good].float()
            logit_bad = circuit_outputs.logits[batch_indices, pos_bad, token_bad].float()
            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()

            kl_loss = total_kl / batch_size_curr

            # Get sparsity loss with adaptive multiplier
            sparsity_loss_dict = circuit_model.get_sparsity_loss(step=total_steps)

            # Apply scheduler's adaptive multiplier
            lambda_mult = scheduler.lambda_multiplier
            sparsity_loss = sparsity_loss_dict['total_sparsity'] * lambda_mult

            # Total loss
            loss = kl_loss * 1.5 + sparsity_loss + task_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1

        # End of epoch
        epoch_duration = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity_loss = epoch_sparsity_loss / len(train_dataloader)

        # Compute current sparsity for display
        current_sparsity = compute_overall_sparsity(circuit_model)

        epoch_pbar.set_postfix({
            'Sp': f"{current_sparsity:.3f}",
            'λ': f"{lambda_mult:.2f}",
            'Phase': scheduler.phase[:4],
            'Time': f"{epoch_duration:.0f}s"
        })

        # Validation and scheduler update every 10 epochs
        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Ep{epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_dataloader,
                device=DEVICE,
                tokenizer=tokenizer,
                verbose=False,
            )

            val_acc = val_results['accuracy']
            val_kl = val_results['kl_div']

            # Update scheduler - this is where the magic happens!
            scheduler.step_update(
                step=total_steps,
                epoch=epoch + 1,
                accuracy=val_acc,
                sparsity_rate=current_sparsity,
                kl_loss=val_kl,
            )

            # Save best checkpoint
            if current_sparsity > scheduler.best_sparsity_at_acceptable_acc:
                _save_checkpoint(
                    circuit_model, optimizer, epoch, total_steps, scheduler,
                    os.path.join(SAVE_DIR, 'best_checkpoint.pt'),
                    GATE_PATTERNS
                )

            circuit_model.train()

            # Check early stopping
            if scheduler.should_stop_early():
                print(f"\n{'='*80}")
                print("  🎉 CONVERGENCE DETECTED - STOPPING EARLY!")
                print(f"{'='*80}")
                summary = scheduler.get_final_summary()
                print(f"\nDiscovered Circuit Summary:")
                print(f"  Final accuracy:  {summary['final_accuracy']:.4f} ({summary['accuracy_drop_pct']:.1f}% drop)")
                print(f"  Final sparsity:  {summary['final_sparsity']:.4f}")
                print(f"  Best sparsity:   {summary['best_sparsity_at_acceptable_acc']:.4f}")
                print(f"  Total epochs:    {summary['total_epochs']}")
                print(f"  Best lambda:     {summary['best_lambda']:.3f}")
                break

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                circuit_model, optimizer, epoch, total_steps, scheduler,
                os.path.join(SAVE_DIR, f'checkpoint_ep{epoch+1}.pt'),
                GATE_PATTERNS
            )

    # --- Save training dynamics plot ---
    print("\n📊 Generating training dynamics visualization...")
    scheduler.plot_training_dynamics(os.path.join(SAVE_DIR, 'fully_adaptive_training.png'))

    # --- Final Evaluation ---
    print(f"\n{'='*80}")
    print("  FINAL EVALUATION")
    print(f"{'='*80}")

    circuit_model.eval()
    print("\n--- Pre-finalization evaluation ---")
    pre_final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Pre-Finalization",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    print("\n--- Analyzing and finalizing circuit ---")
    final_analysis = analyze_and_finalize_circuit(circuit_model)

    print("\n--- Post-finalization evaluation ---")
    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Final Circuit",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    # --- Final Summary ---
    summary = scheduler.get_final_summary()
    final_sparsity = compute_overall_sparsity(circuit_model)

    print(f"\n{'='*80}")
    print("  🎯 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n📊 Accuracy Metrics:")
    print(f"  Baseline:          {base_accuracy:.4f}")
    print(f"  Pre-finalization:  {pre_final_results['accuracy']:.4f} ({(base_accuracy - pre_final_results['accuracy']) / base_accuracy * 100:.1f}% drop)")
    print(f"  Post-finalization: {final_results['accuracy']:.4f} ({(base_accuracy - final_results['accuracy']) / base_accuracy * 100:.1f}% drop)")

    print(f"\n✂️  Sparsity Metrics:")
    print(f"  Final sparsity:    {final_sparsity:.4f} ({final_sparsity * 100:.1f}% pruned)")
    print(f"  Best during train: {summary['best_sparsity_at_acceptable_acc']:.4f}")

    print(f"\n⚙️  Training Stats:")
    print(f"  Total epochs:      {summary['total_epochs']}")
    print(f"  Best lambda:       {summary['best_lambda']:.3f}")
    print(f"  Final phase:       {summary['phase']}")

    if 'prunable_compression' in final_analysis:
        comp = final_analysis['prunable_compression']
        print(f"\n🗜️  Compression:")
        print(f"  Prunable params:   {comp['compression_ratio']:.2f}x compression")
        print(f"  Overall model:     {comp['effective_compression']:.2f}x compression")

    print(f"\n📈 Plots saved to:")
    print(f"  {os.path.join(SAVE_DIR, 'fully_adaptive_training.png')}")

    print(f"\n{'='*80}")
    print("  ✅ DONE! Circuit discovered automatically.")
    print(f"{'='*80}\n")
