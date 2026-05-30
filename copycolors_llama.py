"""
CopyColors MCQA Circuit Discovery Experiment for Llama Models.

Uses dual-stream (clean/corrupted) forward pass with differentiable pruning
via HardConcreteGates to discover the minimal circuit for the CopyColors
in-context learning MCQA task from MIB-bench (ICML 2025).

Task: Given colored object descriptions, answer "What color is X?" from
multiple choices. Tests in-context learning and color-object binding.

Usage:
    python copycolors_llama.py
    python copycolors_llama.py --dry-run
    python copycolors_llama.py --model meta-llama/Llama-3.1-8B --num-choices 4
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

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from dataset.copycolors_llama import (
    CopyColorsDatasetLlama,
    load_copycolors_data,
    run_evaluation,
    filter_dataset_by_model_correctness,
)
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION (tuned for Llama 3.2-1B)
# ==============================================================================
from dataclasses import dataclass


@dataclass
class CopyColorsPruningConfig(PruningConfig):
    """PruningConfig with defaults tuned for CopyColors MCQA on Llama."""
    init_value: float = 0.5
    sparsity_warmup_steps: int = 500
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


# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def _save_checkpoint(circuit_model, optimizer, epoch, total_steps,
                     best_val_accuracy, val_results, path, gate_patterns):
    """Save only gate parameters plus optimizer state."""
    gate_state = {}
    for name, param in circuit_model.named_parameters():
        if any(p in name for p in gate_patterns):
            gate_state[name] = param.data.cpu()

    checkpoint = {
        'epoch': epoch,
        'total_steps': total_steps,
        'gate_state_dict': gate_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'val_results': val_results,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path} (epoch {epoch+1})")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CopyColors MCQA Circuit Discovery for Llama")
    parser.add_argument('--dry-run', action='store_true', help='Quick test with minimal data')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='HuggingFace model name/path')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lambda-sparsity', type=float, default=0.70,
                        help='Sparsity loss weight')
    parser.add_argument('--num-choices', type=int, default=4,
                        help='Number of MCQA answer choices (2-10)')
    parser.add_argument('--hf-token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--save-dir', type=str, default='checkpoints_llama_copycolors',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh even if checkpoints exist')
    args = parser.parse_args()

    # --- Read HF token ---
    hf_token = args.hf_token
    if hf_token is None:
        token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_tokken.txt")
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                hf_token = f.read().strip()
            print(f"Loaded HF token from {token_file}")

    # --- Configuration ---
    MODEL_NAME = args.model
    NUM_EPOCHS = 2 if args.dry_run else args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = 128  # CopyColors prompts are longer than IOI
    LAMBDA_SPARSITY = args.lambda_sparsity
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = args.save_dir

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Dry run settings
    NUM_TRAIN = 20 if args.dry_run else 500
    NUM_VAL = 10 if args.dry_run else 200
    NUM_TEST = 10 if args.dry_run else 500

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Num choices: {args.num_choices}")
    print(f"Lambda sparsity: {LAMBDA_SPARSITY}")
    print(f"Dry run: {args.dry_run}")
    print(f"Save dir: {SAVE_DIR}")

    pruning_config = CopyColorsPruningConfig()

    # --- Model and Tokenizer Setup ---
    print("\n--- Loading tokenizer and models ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}

    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL_NAME, pruning_config, **model_kwargs
    ).to(DEVICE).eval()

    full_model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, **model_kwargs
    ).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    # --- Disable dropout ---
    print("\n--- Disabling all dropout layers in the circuit model ---")
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

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\nSetting up CopyColors MCQA dataset...")

    train_data = load_copycolors_data(
        split="train", num_samples=NUM_TRAIN, num_choices=args.num_choices
    )
    val_data = load_copycolors_data(
        split="validation", num_samples=NUM_VAL, num_choices=args.num_choices
    )
    test_data = load_copycolors_data(
        split="test", num_samples=NUM_TEST, num_choices=args.num_choices
    )

    # Filter datasets by model correctness
    print("\n--- Filtering datasets based on Base Model correctness ---")
    val_data = filter_dataset_by_model_correctness(
        val_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE
    )
    test_data = filter_dataset_by_model_correctness(
        test_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE
    )

    # Create Dataset objects
    train_dataset = CopyColorsDatasetLlama(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = CopyColorsDatasetLlama(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = CopyColorsDatasetLlama(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Baseline Evaluation ---
    print("\n--- Baseline evaluation on full model ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline Full Model",
        full_model_for_faithfulness=None,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )
    base_accuracy = baseline_results.get("accuracy", 0.0)
    base_logit_diff = baseline_results.get("logit_diff", 0.0)

    # --- Initial Circuit Model Evaluation ---
    print("\n--- Initial evaluation of the Circuit Discovery Model ---")
    circuit_model.eval()
    initial_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Initial Circuit Model",
        full_model_for_faithfulness=full_model,
        dataloader=val_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)

    print(f"\n--- Starting training to find 'CopyColors MCQA' circuit ---")
    print(f"Target: Maintain accuracy near baseline ({base_accuracy:.4f})")

    # --- Auto-resume from latest checkpoint ---
    start_epoch = 0
    total_steps = 0
    best_val_accuracy = 0.0

    if not args.no_resume:
        resume_path = None
        if os.path.exists(SAVE_DIR):
            candidates = []
            for f in os.listdir(SAVE_DIR):
                if f.endswith('.pt'):
                    fpath = os.path.join(SAVE_DIR, f)
                    candidates.append((os.path.getmtime(fpath), fpath, f))
            if candidates:
                candidates.sort(reverse=True)
                resume_path = candidates[0][1]
                print(f"\n--- Found checkpoint: {candidates[0][2]} ---")

        if resume_path and os.path.exists(resume_path):
            print(f"--- Auto-resuming from: {resume_path} ---")
            checkpoint = torch.load(resume_path, map_location=DEVICE)

            gate_state = checkpoint['gate_state_dict']
            model_state = circuit_model.state_dict()
            model_state.update(gate_state)
            circuit_model.load_state_dict(model_state)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            total_steps = checkpoint.get('total_steps', 0)
            best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)

            print(f"  Resumed from epoch {start_epoch}, step {total_steps}")
            print(f"  Best val accuracy so far: {best_val_accuracy:.4f}")
        else:
            print("\n--- No checkpoint found, starting fresh ---")

    # --- Pre-cache full model outputs ---
    print("\nPre-caching full model outputs for training data...")
    cached_train_logits = {}
    full_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Caching full model")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)
            out = full_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )
            cached_train_logits[batch_idx] = out.logits.detach()
    print(f"Cached {len(cached_train_logits)} batches of full model outputs.")

    circuit_model.train()

    epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS), desc="Training Progress",
                      initial=start_epoch, total=NUM_EPOCHS)

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0
        epoch_task_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)

            # Forward pass (dual-stream)
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )

            # Use pre-cached full model outputs
            target_logits = cached_train_logits[batch_idx]

            # Calculate KL loss at prediction position
            batch_size_curr = circuit_outputs.logits.size(0)
            total_kl = 0

            for i in range(batch_size_curr):
                pred_pos = batch['prefix_length'][i].item() - 1
                valid_length = batch['attention_mask'][i].sum().item()

                if pred_pos < valid_length and pred_pos < circuit_outputs.logits.size(1):
                    circuit_logits_slice = circuit_outputs.logits[i, pred_pos].float()
                    target_logits_slice = target_logits[i, pred_pos].float()

                    kl = F.kl_div(
                        F.log_softmax(circuit_logits_slice, dim=-1),
                        F.log_softmax(target_logits_slice, dim=-1),
                        reduction='sum',
                        log_target=True,
                    )
                    total_kl += kl

            # Task loss: correct answer letter should beat distractor
            batch_indices = torch.arange(batch_size_curr, device=DEVICE)
            pred_positions = batch['prefix_length'] - 1

            logit_good = circuit_outputs.logits[
                batch_indices, pred_positions, batch['target_token']
            ].float()
            logit_bad = circuit_outputs.logits[
                batch_indices, pred_positions, batch['distractor_token']
            ].float()
            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()

            kl_loss = total_kl / batch_size_curr
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']

            # Total loss
            loss = (1 - LAMBDA_SPARSITY) * (kl_loss + task_loss) + LAMBDA_SPARSITY * sparsity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            optimizer.step()

            # Track losses
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_task_loss += task_loss.item()
            total_steps += 1

        # End of epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        avg_loss = epoch_loss / max(len(train_dataloader), 1)
        avg_kl = epoch_kl_loss / max(len(train_dataloader), 1)
        avg_sparsity = epoch_sparsity_loss / max(len(train_dataloader), 1)

        epoch_pbar.set_postfix({
            'L': f"{avg_loss:.3f}",
            'Sp': f"{avg_sparsity:.3f}",
            'Time': f"{epoch_duration:.2f}s"
        })

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Val Ep {epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_dataloader,
                device=DEVICE,
                tokenizer=tokenizer,
            )

            if (epoch + 1) % 50 == 0:
                test_results = run_evaluation(
                    model_to_eval=circuit_model,
                    model_name=f"Test Ep {epoch+1}",
                    full_model_for_faithfulness=full_model,
                    dataloader=test_dataloader,
                    device=DEVICE,
                    tokenizer=tokenizer,
                )

            val_acc = val_results.get('accuracy', 0.0)
            # Only track best checkpoint AFTER warmup ends, otherwise we just
            # save the unpruned model (which has 100% accuracy but 0% pruning)
            warmup_done = total_steps > pruning_config.sparsity_warmup_steps
            if warmup_done and val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                _save_checkpoint(
                    circuit_model, optimizer, epoch, total_steps,
                    best_val_accuracy, val_results,
                    os.path.join(SAVE_DIR, 'best_checkpoint.pt'),
                    GATE_PATTERNS,
                )
                print(f"  >> New best val accuracy: {best_val_accuracy:.4f} (saved, post-warmup)")

            circuit_model.train()

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                circuit_model, optimizer, epoch, total_steps,
                best_val_accuracy, None,
                os.path.join(SAVE_DIR, f'checkpoint_ep{epoch+1}.pt'),
                GATE_PATTERNS,
            )

    # Save final checkpoint
    if start_epoch < NUM_EPOCHS:
        _save_checkpoint(
            circuit_model, optimizer, epoch, total_steps,
            best_val_accuracy, None,
            os.path.join(SAVE_DIR, 'final_checkpoint.pt'),
            GATE_PATTERNS,
        )

    # --- Pre-finalization evaluation ---
    print("\n--- Pre-finalization evaluation on test set ---")
    circuit_model.eval()
    pre_final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Pre-Finalization (Test Set)",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    # --- Final Analysis and Pruning ---
    print("\n--- Analyzing and finalizing circuit ---")
    analyze_and_finalize_circuit(circuit_model)

    # --- Final Evaluation on Test Set ---
    print("\n--- Final evaluation on test set ---")
    circuit_model.eval()
    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Final Pruned Circuit",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    # --- Best checkpoint evaluation ---
    best_ckpt_path = os.path.join(SAVE_DIR, 'best_checkpoint.pt')
    if os.path.exists(best_ckpt_path):
        print("\n" + "=" * 80)
        print("  EVALUATING BEST CHECKPOINT (full post-pruning pipeline)")
        print("=" * 80)

        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        gate_state = best_ckpt['gate_state_dict']
        model_state = circuit_model.state_dict()
        model_state.update(gate_state)
        circuit_model.load_state_dict(model_state)

        circuit_model.set_final_circuit_mode(False)

        circuit_model.eval()
        print("\n--- Best checkpoint: Pre-finalization test eval ---")
        best_pre_final = run_evaluation(
            model_to_eval=circuit_model,
            model_name="Best Ckpt Pre-Finalization",
            full_model_for_faithfulness=full_model,
            dataloader=test_dataloader,
            device=DEVICE,
            tokenizer=tokenizer,
        )

        print("\n--- Best checkpoint: Analyzing and finalizing circuit ---")
        analyze_and_finalize_circuit(circuit_model)

        print("\n--- Best checkpoint: Post-finalization test eval ---")
        circuit_model.eval()
        best_post_final = run_evaluation(
            model_to_eval=circuit_model,
            model_name="Best Ckpt Post-Finalization",
            full_model_for_faithfulness=full_model,
            dataloader=test_dataloader,
            device=DEVICE,
            tokenizer=tokenizer,
        )

        print(f"\n  Best checkpoint saved at epoch: {best_ckpt['epoch']+1}")
        print(f"  Pre-finalization accuracy:  {best_pre_final['accuracy']:.4f}")
        print(f"  Post-finalization accuracy: {best_post_final['accuracy']:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - CopyColors MCQA Circuit Discovery")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Num choices: {args.num_choices}")
    print(f"Lambda sparsity: {LAMBDA_SPARSITY}")
    print(f"Baseline Accuracy: {base_accuracy:.4f}")
    print(f"Baseline Logit Diff: {base_logit_diff:.4f}")
    print(f"Final Circuit Accuracy: {final_results['accuracy']:.4f} (drop: {base_accuracy - final_results['accuracy']:.4f})")
    print(f"Final Circuit Logit Diff: {final_results['logit_diff']:.4f}")
    print(f"Final KL Divergence: {final_results['kl_div']:.4f}")
    print(f"Exact Match Rate: {final_results['exact_match']:.4f}")

    sparsity_stats = circuit_model.get_sparsity_loss(step=total_steps)
    print(f"\nSparsity Statistics:")
    for key, value in sparsity_stats.items():
        if key != 'total_sparsity':
            print(f"  - {key}: {value:.4f}")
    print("=" * 60)
