import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import random
from models.gpt2_circuit import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, GPT2LMHeadModel, PruningConfig
from dataset.gp import GPDataset, load_or_generate_gp_data, run_evaluation, filter_dataset_by_model_correctness

import torch
import torch.nn as nn
from tqdm import tqdm
from models.l0 import HardConcreteGate

import torch
import torch.nn as nn
from tqdm import tqdm
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION
# ==============================================================================
from dataclasses import dataclass
# PRUNING_FACTOR = 1.0

# # @dataclass
# @dataclass
# class PruningConfig:
#     init_value: float = 1.0
#     sparsity_warmup_steps: int = 0

#     # --- Fine-grained pruning (existing) ---
#     # Attention Head Pruning
#     prune_attention_heads: bool = True
#     lambda_attention_heads: float = 0.02 * PRUNING_FACTOR

#     # MLP neuron pruning
#     prune_mlp_hidden: bool = True
#     lambda_mlp_hidden: float = 0.00005 * PRUNING_FACTOR
#     prune_mlp_output: bool = True
#     lambda_mlp_output: float = 0.00005 * PRUNING_FACTOR
    
    
#     prune_attention_neurons: bool = True
#     lambda_attention_neurons: float = 0.0002 * PRUNING_FACTOR
    
#     prune_embedding: bool = False
#     lambda_embedding: float = 1 * PRUNING_FACTOR
    
#     # Prune entire attention blocks
#     prune_attention_blocks: bool = True
#     lambda_attention_blocks: float = 0.000005 * PRUNING_FACTOR
    
#     # Prune entire MLP blocks
#     prune_mlp_blocks: bool = True
#     lambda_mlp_blocks: float = 0.05 * PRUNING_FACTOR
    
#     # Prune entire transformer layers
#     prune_full_layers: bool = False
#     lambda_full_layers: float = 0.0000005 * PRUNING_FACTOR




# PRUNING_FACTOR = 1.0

# # @dataclass
# @dataclass
# class PruningConfig:
#     init_value: float = 1.0
#     sparsity_warmup_steps: int = 0

#     # --- Fine-grained pruning (existing) ---
#     # Attention Head Pruning
#     prune_attention_heads: bool = True
#     lambda_attention_heads: float = 0.045 * PRUNING_FACTOR # 0.027 * PRUNING_FACTOR

#     # MLP neuron pruning
#     prune_mlp_hidden: bool = True
#     lambda_mlp_hidden: float = 0.0005 * PRUNING_FACTOR
#     prune_mlp_output: bool = True
#     lambda_mlp_output: float = 0.0005 * PRUNING_FACTOR
    
    
#     prune_attention_neurons: bool = True
#     lambda_attention_neurons: float = 0.0002 * PRUNING_FACTOR
    
#     prune_embedding: bool = False
#     lambda_embedding: float = 1 * PRUNING_FACTOR
    
#     # Prune entire attention blocks
#     prune_attention_blocks: bool = True
#     lambda_attention_blocks: float = 0.000005 * PRUNING_FACTOR
    
#     # Prune entire MLP blocks
#     prune_mlp_blocks: bool = True
#     lambda_mlp_blocks: float = 0.6 * PRUNING_FACTOR
    
#     # Prune entire transformer layers
#     prune_full_layers: bool = False
#     lambda_full_layers: float = 0.000000005 * PRUNING_FACTOR


# PRUNING_FACTOR = 0.01  # Keep this at 1.0 to keep math simple

# @dataclass
# class PruningConfig:
#     # Start with gates FULLY OPEN (log_alpha > 0) so gradient flows immediately
#     init_value: float = 0.5 
    
#     # CRITICAL: Don't prune for the first ~5-10 epochs
#     sparsity_warmup_steps: int = 1000 

#     # --- Lambdas ---
#     # These values are tuned for GPT-2 Small scale.
#     # If a lambda is too high, the gate dies instantly (instability).
#     # If too low, it never closes.
    
#     depth_penalty_scaling: float = 0.1
    
#     # 1. Heads: Moderate cost. We want to remove many, but they are useful.
#     prune_attention_heads: bool = True
#     lambda_attention_heads: float = 0.8 

#     # 2. Neurons (Hidden): There are 3072 of them. 
#     # Individual neurons are weak. The penalty must be small, or you kill them all.
#     prune_mlp_hidden: bool = True
#     lambda_mlp_hidden: float = 1.0  # Much lower than 25.0!

#     # 3. MLP Output (Residual): This is a "strong" cut.
#     prune_mlp_output: bool = True
#     lambda_mlp_output: float = 1.0 
    
#     # 4. Attention Neurons: 
#     prune_attention_neurons: bool = True
#     lambda_attention_neurons: float = 0.15

#     # Structure pruning (Blocks/Layers)
#     # Usually easier to prune fine-grained first, then structure.
#     prune_attention_blocks: bool = True
#     lambda_attention_blocks: float = 0.5
    
#     prune_mlp_blocks: bool = True
#     lambda_mlp_blocks: float = 0.5 
    
#     prune_full_layers: bool = False
#     lambda_full_layers: float = 0.0
    
#     prune_embedding: bool = False
#     lambda_embedding: float = 1 * PRUNING_FACTOR


PRUNING_FACTOR = 1.0  # Keep this at 1.0 to keep math simple

@dataclass
class PruningConfig:
    # Start with gates FULLY OPEN (log_alpha > 0) so gradient flows immediately
    init_value: float = 0.5 
    
    # CRITICAL: Don't prune for the first ~5-10 epochs
    sparsity_warmup_steps: int = 1000 

    # --- Lambdas ---
    # These values are tuned for GPT-2 Small scale.
    # If a lambda is too high, the gate dies instantly (instability).
    # If too low, it never closes.
    
    depth_penalty_scaling: float = 0.0
    
    # 1. Heads: Moderate cost. We want to remove many, but they are useful.
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0 

    # 2. Neurons (Hidden): There are 3072 of them. 
    # Individual neurons are weak. The penalty must be small, or you kill them all.
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0  # Much lower than 25.0!

    # 3. MLP Output (Residual): This is a "strong" cut.
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0 
    
    # 4. Attention Neurons: 
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0

    # Structure pruning (Blocks/Layers)
    # Usually easier to prune fine-grained first, then structure.
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0
    
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 2.0 
    
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0
    
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR
# ==============================================================================
# MAIN EXECUTION FOR GENDER PRONOUNS TASK
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 500
    LEARNING_RATE = 3e-1
    BATCH_SIZE = 64  # Matching the reference implementation
    MAX_SEQ_LEN = 32
    ACCURACY_BUDGET = 0.05  # Allow 5% accuracy drop from baseline
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruning_config = PruningConfig()
    
    # --- Model and Tokenizer Setup ---
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE).eval()
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters(): param.requires_grad = False

    # ----- Disable all built-in dropout layers in the circuit model ---
    print("\n--- Disabling all built-in dropout layers in the circuit model ---")
    disable_dropout(circuit_model)
    # -----------------------------------------------------------------
    
    # --- Freeze the base model and unfreeze only the gates ---
    print("Freezing base model weights and unfreezing gate parameters...")
    total_params = 0
    trainable_params = 0
    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        if 'gate' not in name:
            # print(f"  Freezing parameter: {name}")
            param.requires_grad = False
        else:
            # print(f"  Unfreezing for training: {name}")
            param.requires_grad = True
            trainable_params += param.numel()
            
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\nSetting up Gender Pronouns dataset...")
    # 1. Load Raw Data (Lists of dictionaries)
    test_data = load_or_generate_gp_data(split="test", num_samples=100000)
    train_data = load_or_generate_gp_data(split="train", num_samples=100000)
    val_data = load_or_generate_gp_data(split="validation", num_samples=10000)

    # 2. Filter the Raw Data Lists (Pass *_data, NOT *_dataset)
    print("\n--- Filtering datasets based on Base Model correctness ---")
    
    # train_data = filter_dataset_by_model_correctness(
    #     train_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE
    # )
    
    val_data = filter_dataset_by_model_correctness(
        val_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE
    )
    
    test_data = filter_dataset_by_model_correctness(
        test_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE
    )

    print(f"\nFinal Train samples: {len(train_data)}")
    print(f"Final Val samples: {len(val_data)}")
    print(f"Final Test samples: {len(test_data)}")
    
    import os
    from datasets import Dataset, DatasetDict

    # ==============================================================================
    # SAVE FILTERED DATASETS (ARROW FORMAT)
    # ==============================================================================
    print("\n--- Saving filtered datasets to Arrow format ---")

    # Define where you want the data saved
    save_path = "./filtered_datasets/gp"

    # 1. Convert the filtered lists (dicts) back into Hugging Face Datasets
    #    (Assuming train_data, val_data, test_data are lists of dictionaries)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    # 2. Combine into a single DatasetDict (optional, but cleaner for loading later)
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # 3. Save to disk
    #    This creates a folder structure containing the Arrow files
    dataset_dict.save_to_disk(save_path)
    
    # Create dataset objects
    train_dataset = GPDataset(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = GPDataset(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = GPDataset(test_data, tokenizer, max_length=MAX_SEQ_LEN)
    
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
    initial_accuracy = initial_results.get("accuracy", 0.0)
    initial_logit_diff = initial_results.get("logit_diff", 0.0)

    # --- Training ---
    # The optimizer will now only see the parameters that require gradients (the gates)
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    print(f"\n--- Starting training to find 'Gender Pronouns' circuit ---")
    print(f"Target: Maintain accuracy within {ACCURACY_BUDGET*100}% of baseline ({base_accuracy:.4f})")
    
    circuit_model.train()
    total_steps = 0
    
    # --- CHANGED: Single tqdm loop over epochs ---
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")
    
    for epoch in epoch_pbar:
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0
        
        # --- CHANGED: Removed inner tqdm ---
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Move batch to device
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
            
            # Calculate loss at the prediction positions
            batch_size = circuit_outputs.logits.size(0)
            total_kl = 0
            
            for i in range(batch_size):
                pred_pos = batch['prefix_length'][i] - 1
                
                circuit_logits = circuit_outputs.logits[i, pred_pos, :]
                target_logits = target_outputs.logits[i, pred_pos, :]
                
                # KL divergence loss
                kl = F.kl_div(
                    F.log_softmax(circuit_logits, dim=-1), 
                    F.log_softmax(target_logits, dim=-1), 
                    reduction='sum', 
                    log_target=True
                )
                total_kl += kl
            
            # Task loss (Commented out in original, kept commented out here)
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
            task_loss = F.relu(0.1 - (logit_good - logit_bad)).mean()
            
            kl_loss = total_kl / batch_size
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            lambda_sparsity = 0.70
            # Total loss
            loss = (1-lambda_sparsity)*(kl_loss + task_loss) + lambda_sparsity * sparsity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1

        # Calculate averages for progress bar
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)
        
        # Update progress bar description
        epoch_pbar.set_postfix({
            'L': f"{avg_loss:.3f}", 
            'KL': f"{avg_kl:.3f}", 
            'Sp': f"{avg_sparsity:.3f}"
        })
        
        # --- CHANGED: Run evaluation every 10 epochs ---
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
    pruning_config.prune_full_layers = True  # Enable full layer pruning for final evaluation
    circuit_model.set_pruning_config(pruning_config)
    # --- Final Evaluation on Test Set ---
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
    print("\n" + "="*60)
    print("FINAL SUMMARY - Gender Pronouns Circuit Discovery")
    print("="*60)
    print(f"Baseline Accuracy: {base_accuracy:.4f}")
    print(f"Baseline Logit Diff: {base_logit_diff:.4f}")
    print(f"Final Circuit Accuracy: {final_results['accuracy']:.4f} (drop: {base_accuracy - final_results['accuracy']:.4f})")
    print(f"Final Circuit Logit Diff: {final_results['logit_diff']:.4f}")
    print(f"Final KL Divergence: {final_results['kl_div']:.4f}")
    print(f"Exact Match Rate: {final_results['exact_match']:.4f}")
    
    # Get sparsity statistics
    sparsity_stats = circuit_model.get_sparsity_loss(step=total_steps)
    print(f"\nSparsity Statistics:")
    for key, value in sparsity_stats.items():
        if key != 'total_sparsity':
            print(f"  - {key}: {value:.4f}")
    print("="*60)