"""
Docstring Task - Circuit Discovery Training Script for GPT-2.

Discovers the minimal circuit in GPT-2 that implements the docstring
argument prediction task from the ACDC paper (Conmy et al., NeurIPS 2023).

Usage:
    python docstring.py

The task: Given a Python function def with :param docstrings, predict the
next argument name. The model sees the function signature and partial
docstring, ending with ":param ", and must predict which argument comes next.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

from models.gpt2_circuit import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, GPT2LMHeadModel, PruningConfig
from dataset.docstring import DocstringDataset, generate_docstring_data, run_evaluation, filter_dataset_by_model_correctness
from utils import disable_dropout, analyze_and_finalize_circuit


# ==============================================================================
# PRUNING CONFIGURATION
# ==============================================================================

PRUNING_FACTOR = 1.0

@dataclass
class PruningConfig:
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
    lambda_mlp_blocks: float = 2.0

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0

    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR


# ==============================================================================
# MAIN EXECUTION FOR DOCSTRING TASK
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 500
    LEARNING_RATE = 3e-2
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 64
    ACCURACY_BUDGET = 0.05
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruning_config = PruningConfig()

    # --- Model and Tokenizer Setup ---
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE).eval()
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    # --- Disable all built-in dropout layers ---
    print("\n--- Disabling all built-in dropout layers in the circuit model ---")
    disable_dropout(circuit_model)

    # --- Freeze base model, unfreeze gates ---
    print("Freezing base model weights and unfreezing gate parameters...")
    total_params = 0
    trainable_params = 0
    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        if 'gate' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params += param.numel()

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\nGenerating Docstring task data...")
    train_data = generate_docstring_data(num_samples=500, seed=42)
    val_data = generate_docstring_data(num_samples=500, seed=123)
    test_data = generate_docstring_data(num_samples=1000, seed=456)

    # Filter by model correctness
    print("\n--- Filtering datasets based on Base Model correctness ---")
    val_data = filter_dataset_by_model_correctness(
        val_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE
    )
    test_data = filter_dataset_by_model_correctness(
        test_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE
    )

    print(f"\nFinal Train samples: {len(train_data)}")
    print(f"Final Val samples: {len(val_data)}")
    print(f"Final Test samples: {len(test_data)}")

    # Create dataset objects
    train_dataset = DocstringDataset(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = DocstringDataset(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = DocstringDataset(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    # Create dataloaders
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
        tokenizer=tokenizer
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
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer
    )

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)

    print(f"\n--- Starting training to find 'Docstring' circuit ---")
    print(f"Target: Maintain accuracy within {ACCURACY_BUDGET*100}% of baseline ({base_accuracy:.4f})")

    circuit_model.train()
    total_steps = 0
    lambda_sparsity = 0.95

    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")

    for epoch in epoch_pbar:
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)

            # Forward pass through circuit model
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Get target outputs from full model
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

            # Calculate KL loss at prediction positions
            batch_size = circuit_outputs.logits.size(0)
            total_kl = 0

            for i in range(batch_size):
                pred_pos = batch['prefix_length'][i] - 1

                circuit_logits = circuit_outputs.logits[i, pred_pos, :]
                target_logits = target_outputs.logits[i, pred_pos, :]

                kl = F.kl_div(
                    F.log_softmax(circuit_logits, dim=-1),
                    F.log_softmax(target_logits, dim=-1),
                    reduction='sum',
                    log_target=True
                )
                total_kl += kl

            # Task loss: margin between correct and max wrong
            logit_good = circuit_outputs.logits[
                torch.arange(batch_size),
                batch['prefix_length'] - 1,
                batch['target_token']
            ]
            logit_bad = circuit_outputs.logits[
                torch.arange(batch_size),
                batch['prefix_length'] - 1,
                batch['distractor_token']
            ]
            task_loss = F.relu(2.0 - (logit_good - logit_bad)).mean()

            kl_loss = total_kl / batch_size
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']

            # Total loss
            loss = (1 - lambda_sparsity) * (kl_loss + task_loss) + lambda_sparsity * sparsity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1

        # Calculate averages
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)

        epoch_pbar.set_postfix({
            'L': f"{avg_loss:.3f}",
            'KL': f"{avg_kl:.3f}",
            'Sp': f"{avg_sparsity:.3f}"
        })

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            print(f"\n--- Validation at Epoch {epoch+1} ---")
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Circuit after Epoch {epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=test_dataloader,
                device=DEVICE,
                tokenizer=tokenizer
            )

            current_accuracy = val_results.get("accuracy", 0.0)
            accuracy_drop = base_accuracy - current_accuracy
            if accuracy_drop > ACCURACY_BUDGET:
                print(f"  WARNING: Accuracy drop ({accuracy_drop:.4f}) exceeds budget ({ACCURACY_BUDGET})!")

            circuit_model.train()

    # --- Final Analysis and Pruning ---
    print("\n--- Analyzing and finalizing circuit ---")
    pruning_config.prune_full_layers = True
    circuit_model.set_pruning_config(pruning_config)
    analyze_and_finalize_circuit(circuit_model)

    print("\n--- Final evaluation on test set ---")
    circuit_model.eval()
    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Final Pruned Circuit (Optimal Thresholds)",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - Docstring Circuit Discovery")
    print("=" * 60)
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
