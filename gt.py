import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import random
from models.gpt2_test_copy import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, GPT2LMHeadModel
from dataset.gt_gpt2 import GTDataset, load_or_generate_gt_data, create_two_digit_token_mapping, run_evaluation, filter_dataset_by_model_correctness


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
from datasets import Dataset, DatasetDict
# PRUNING_FACTOR = 0.9


# @dataclass
# class PruningConfig:
#     init_value: float = 1
#     sparsity_warmup_steps: int = 0

#     # --- Fine-grained pruning (existing) ---
#     # Attention Head Pruning
#     prune_attention_heads: bool = True
#     lambda_attention_heads: float = 0.07 * PRUNING_FACTOR

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
#     lambda_attention_blocks: float = 0.05 * PRUNING_FACTOR
    
#     # Prune entire MLP blocks
#     prune_mlp_blocks: bool = True
#     lambda_mlp_blocks: float = 0.5 * PRUNING_FACTOR
    
#     # Prune entire transformer layers
#     prune_full_layers: bool = False
#     lambda_full_layers: float = 0.05 * PRUNING_FACTOR



PRUNING_FACTOR = 0.001  # Keep this at 1.0 to keep math simple

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
    lambda_attention_heads: float = 0.0

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
    lambda_mlp_blocks: float = 1.0 
    
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0
    
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR
# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    # (Same as before)
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 250
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 16
    MAX_SEQ_LEN = 32
    PROB_DIFF_BUDGET = 0.2
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
            param.requires_grad = False
        else:
            # print(f"  Unfreezing for training: {name}")
            param.requires_grad = True
            trainable_params += param.numel()
            
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")
    two_digit_tokens = create_two_digit_token_mapping(tokenizer)
    
    # --- Dataset Setup ---
    print("\nSetting up dataset...")
    # Load from disk with fallback to generation
    train_data = load_or_generate_gt_data(split="train", num_samples=200)
    val_data = load_or_generate_gt_data(split="validation", num_samples=200)
    test_data = load_or_generate_gt_data(split="test", num_samples=1000)

    # Filter datasets to keep only samples where the full model is correct
    train_data = filter_dataset_by_model_correctness(train_data, full_model, tokenizer, DEVICE, two_digit_tokens)
    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, DEVICE, two_digit_tokens)
    test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, DEVICE, two_digit_tokens)
    
    import os
    from datasets import Dataset, DatasetDict

    # ==============================================================================
    # SAVE FILTERED DATASETS (ARROW FORMAT)
    # ==============================================================================
    print("\n--- Saving filtered datasets to Arrow format ---")

    # Define where you want the data saved
    save_path = "./filtered_datasets/gt"

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

    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
    # Ensure all splits have at least 1 example
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError("One of the dataset splits is empty. Please check your data generation or loading process.")

    # Create dataset objects
    train_dataset = GTDataset(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = GTDataset(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = GTDataset(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Baseline Evaluation ---
    print("\n--- Test Loader ---")
    baseline_results = run_evaluation(model_to_eval=full_model, model_name="Baseline Full Model", full_model_for_faithfulness=None, dataloader=test_dataloader, device=DEVICE, two_digit_tokens=two_digit_tokens, tokenizer=tokenizer)
    
    # Optional: Baseline on Val/Train if needed
    # print("Validation Results:")
    # run_evaluation(model_to_eval=full_model, model_name="Baseline Full Model", ...)
    
    base_prob_diff = baseline_results.get("prob_diff", 0.0)
    
    # --- Test Circuit model ---
    print("\n--- Initial evaluation of the Circuit Discovery Model ---")
    circuit_model.eval()
    initial_results = run_evaluation(model_to_eval=circuit_model, model_name="Initial Circuit Model", full_model_for_faithfulness=full_model, dataloader=val_dataloader, device=DEVICE, two_digit_tokens=two_digit_tokens)
    initial_prob_diff = initial_results.get("prob_diff", 0.0)

    # --- Training ---
    # The optimizer will now only see the parameters that require gradients (the gates)
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    print(f"\n--- Starting training to find 'Greater-Than' circuit ---")
    circuit_model.train()
    total_steps = 0
    
    # Pre-compute token mapping tensors
    sorted_tokens = sorted(two_digit_tokens.items()) 
    sorted_nums = [item[0] for item in sorted_tokens] 
    num_to_idx = {num: i for i, num in enumerate(sorted_nums)} 
    digit_token_ids = torch.tensor([item[1] for item in sorted_tokens], device=DEVICE) 

    # --- CHANGED: Single tqdm loop over epochs ---
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")

    for epoch in epoch_pbar:
        epoch_loss = 0
        epoch_kl = 0
        epoch_sparsity = 0
        
        # --- CHANGED: Removed inner tqdm ---
        for batch in train_dataloader:
            optimizer.zero_grad()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor): batch[key] = val.to(DEVICE)
            
            circuit_outputs = circuit_model(
                input_ids=batch['clean_input_ids'], 
                corrupted_input_ids=batch['corrupted_input_ids'], 
                attention_mask=batch['clean_attention_mask']
            )
            
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=batch['clean_input_ids'], 
                    attention_mask=batch['clean_attention_mask']
                )

            last_token_circuit_logits = circuit_outputs.logits[torch.arange(circuit_outputs.logits.size(0)), batch['last_token_idx'], :]
            last_token_target_logits = target_outputs.logits[torch.arange(target_outputs.logits.size(0)), batch['last_token_idx'], :]
            
            digit_logits_circuit = torch.gather( 
                last_token_circuit_logits,
                1, 
                digit_token_ids.unsqueeze(0).expand(last_token_circuit_logits.shape[0], -1)
            )
            digit_logits_target = torch.gather(
                last_token_target_logits,
                1, 
                digit_token_ids.unsqueeze(0).expand(last_token_target_logits.shape[0], -1)
            )  
            
            kl_loss = F.kl_div(
                F.log_softmax(digit_logits_circuit, dim=-1), 
                F.log_softmax(digit_logits_target, dim=-1), 
                reduction='batchmean', 
                log_target=True
            )
            
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            lambda_sparsity = 0.90
            loss = (1-lambda_sparsity)*(kl_loss) + lambda_sparsity * sparsity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            optimizer.step()
            total_steps += 1
            
            # Track accumulation
            epoch_loss += loss.item()
            epoch_kl += kl_loss.item() # Tracking raw KL
            epoch_sparsity += sparsity_loss.item()
        
        # Update progress bar
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl / len(train_dataloader)
        avg_sp = epoch_sparsity / len(train_dataloader)
        
        epoch_pbar.set_postfix({
            'L': f"{avg_loss:.3f}", 
            'KL': f"{avg_kl:.3f}", 
            'Sp': f"{avg_sp:.3f}"
        })

        # --- CHANGED: Run evaluation every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            print(f"\n--- Validation at Epoch {epoch+1} ---")
            run_evaluation(
                model_to_eval=circuit_model, 
                model_name=f"Circuit after Epoch {epoch+1}", 
                full_model_for_faithfulness=full_model, 
                dataloader=val_dataloader, 
                device=DEVICE, 
                two_digit_tokens=two_digit_tokens
            )
            circuit_model.train()

    # --- Finalize ---
    analyze_and_finalize_circuit(circuit_model)
    
    final_results = run_evaluation(
        model_to_eval=circuit_model, 
        model_name="Final Pruned Circuit (Optimal Thresholds)", 
        full_model_for_faithfulness=full_model, 
        dataloader=test_dataloader, 
        device=DEVICE, 
        two_digit_tokens=two_digit_tokens
    )
    
    print("Baseline Results:", baseline_results)