import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import random
import time
from models.gpt2_circuit import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, GPT2LMHeadModel, PruningConfig
from dataset.ioi import IOIDataset, load_or_generate_ioi_data, run_evaluation, filter_dataset_by_model_correctness
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION
# ==============================================================================
from dataclasses import dataclass

PRUNING_FACTOR = 25
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
    
    depth_penalty_scaling: float = 0.1
    
    # 1. Heads: Moderate cost. We want to remove many, but they are useful.
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.8 

    # 2. Neurons (Hidden): There are 3072 of them. 
    # Individual neurons are weak. The penalty must be small, or you kill them all.
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0  # Much lower than 25.0!

    # 3. MLP Output (Residual): This is a "strong" cut.
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0 
    
    # 4. Attention Neurons: 
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.15

    # Structure pruning (Blocks/Layers)
    # Usually easier to prune fine-grained first, then structure.
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.5
    
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.5 
    
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0
    
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR

# ==============================================================================
# MAIN EXECUTION FOR IOI TASK
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 500

    LEARNING_RATE = 3e-2
    BATCH_SIZE = 32 
    MAX_SEQ_LEN = 32
    ACCURACY_BUDGET = 0.05 
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
    
    # --- Freeze the base model and unfreeze only the gates ---
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
    print("\nSetting up IOI dataset...")
    
    # 1. Load Raw Data
    test_data = load_or_generate_ioi_data(split="test", num_samples=1000) 
    train_data = load_or_generate_ioi_data(split="train", num_samples=200)
    val_data = load_or_generate_ioi_data(split="validation", num_samples=200)

    # 2. Filter datasets
    print("\n--- Filtering datasets based on Base Model correctness ---")
    # train_data = filter_dataset_by_model_correctness(train_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)
    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)
    test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)

    # 3. Create Final Dataset Objects
    train_dataset = IOIDataset(train_data, tokenizer)
    val_dataset = IOIDataset(val_data, tokenizer)
    test_dataset = IOIDataset(test_data, tokenizer)

    # 4. Create DataLoaders
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
        dataloader=val_dataloader, 
        device=DEVICE, 
        tokenizer=tokenizer
    )

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    print(f"\n--- Starting training to find 'Indirect Object Identification' circuit ---")
    print(f"Target: Maintain accuracy within {ACCURACY_BUDGET*100}% of baseline ({base_accuracy:.4f})")

    circuit_model.train()
    total_steps = 0
    
    # --- CHANGED: Outer tqdm for epochs ---
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")
    
    for epoch in epoch_pbar:
        # --- CHANGED: Timer start inside epoch ---
        epoch_start_time = time.time()
        
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
            
            # Forward pass
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Get target outputs
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            
            # Calculate KL loss
            batch_size_curr = circuit_outputs.logits.size(0)
            total_kl = 0
            
            for i in range(batch_size_curr):
                t_start = batch['T_Start'][i].item()-1
                t_end = batch['T_End'][i].item()-1
                
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, valid_length)
                
                if t_start < end_pos:
                    circuit_logits = circuit_outputs.logits[i, t_start:end_pos]
                    target_logits = target_outputs.logits[i, t_start:end_pos]
                    
                    kl = F.kl_div(
                        F.log_softmax(circuit_logits, dim=-1),
                        F.log_softmax(target_logits, dim=-1),
                        reduction='sum',
                        log_target=True
                    )
                    total_kl += kl
            
            # Task loss calculation
            pos_good = batch['T_Start'] - 1
            pos_bad = batch['D_Start'] - 1
            token_good = batch['target_tokens'][:, 0]
            token_bad = batch['distractor_tokens'][:, 0]
            batch_indices = torch.arange(batch_size_curr, device=DEVICE)
            
            logit_good = circuit_outputs.logits[batch_indices, pos_good, token_good]
            logit_bad = circuit_outputs.logits[batch_indices, pos_bad, token_bad]

            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()
            
            kl_loss = total_kl / batch_size_curr
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            # Total loss
            loss = kl_loss*1.5 + sparsity_loss + task_loss
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1
        
        # End of epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Averages
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)

        # Update progress bar
        epoch_pbar.set_postfix({
            'L': f"{avg_loss:.3f}", 
            'Sp': f"{avg_sparsity:.3f}",
            'Time': f"{epoch_duration:.2f}s"
        })
        
        # --- CHANGED: Validation every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            # print(f"\n--- Validation evaluation after epoch {epoch+1} ---")
            circuit_model.eval()
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Val Ep {epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_dataloader,
                device=DEVICE,
                tokenizer=tokenizer
            )
            circuit_model.train()
    
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
        tokenizer=tokenizer
    )