import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import random
from models.gpt2_test import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, GPT2LMHeadModel
from dataset.gt_gpt2 import GTDataset, load_or_generate_gt_data, create_two_digit_token_mapping, run_evaluation


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
PRUNING_FACTOR = 4


# @dataclass
@dataclass
class PruningConfig:
    init_value: float = 1
    sparsity_warmup_steps: int = 0

    # --- Fine-grained pruning (existing) ---
    # Attention Head Pruning
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.001 * PRUNING_FACTOR

    # MLP neuron pruning
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 0.00005 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 0.00005 * PRUNING_FACTOR
    
    
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.0002 * PRUNING_FACTOR
    
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR
    
    # Prune entire attention blocks
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.05 * PRUNING_FACTOR
    
    # Prune entire MLP blocks
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.05 * PRUNING_FACTOR
    
    # Prune entire transformer layers
    prune_full_layers: bool = True
    lambda_full_layers: float = 0.05 * PRUNING_FACTOR
# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    # (Same as before)
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-3
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 64
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
            print(f"  Unfreezing for training: {name}")
            param.requires_grad = True
            trainable_params += param.numel()
            
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\nSetting up dataset...")
    # Load from disk with fallback to generation
    train_data = load_or_generate_gt_data(split="train")
    val_data = load_or_generate_gt_data(split="validation")
    test_data = load_or_generate_gt_data(split="test")
    
    # combine train and validation and test and split again
    data = train_data + val_data + test_data
    random.shuffle(data)
    train_data = data[:int(0.5 * len(data))]
    val_data = data[int(0.5 * len(data)):int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]
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

    # Create token mapping
    two_digit_tokens = create_two_digit_token_mapping(tokenizer)


    # --- Baseline Evaluation ---
    baseline_results = run_evaluation(model_to_eval=full_model, model_name="Baseline Full Model", full_model_for_faithfulness=None, dataloader=val_dataloader, device=DEVICE, two_digit_tokens=two_digit_tokens, tokenizer=tokenizer)
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
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor): batch[key] = val.to(DEVICE)
            
            circuit_outputs = circuit_model(input_ids=batch['clean_input_ids'], corrupted_input_ids=batch['corrupted_input_ids'], attention_mask=batch['clean_attention_mask'])
            with torch.no_grad():
                target_outputs = full_model(input_ids=batch['clean_input_ids'], attention_mask=batch['clean_attention_mask'])

            last_token_circuit_logits = circuit_outputs.logits[torch.arange(circuit_outputs.logits.size(0)), batch['last_token_idx'], :]
            last_token_target_logits = target_outputs.logits[torch.arange(target_outputs.logits.size(0)), batch['last_token_idx'], :]

            kl_loss = F.kl_div(F.log_softmax(last_token_circuit_logits, dim=-1), F.log_softmax(last_token_target_logits, dim=-1), reduction='batchmean', log_target=True)
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            kl_loss = kl_loss *2
            loss = kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            total_steps += 1
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss.item():.4f} | KL Loss: {kl_loss.item():.4f} | Sparsity Loss: {sparsity_loss.item():.4f}")
        # --- Epoch Validation ---
        run_evaluation(model_to_eval=circuit_model, model_name=f"Circuit after Epoch {epoch+1}", full_model_for_faithfulness=full_model, dataloader=val_dataloader, device=DEVICE, two_digit_tokens=two_digit_tokens)
        circuit_model.train()
        
        #test main model
        #run_evaluation(model_to_eval=full_model, model_name=f"Full Model after Epoch {epoch+1}", full_model_for_faithfulness=full_model, dataloader=val_dataloader, device=DEVICE, two_digit_tokens=two_digit_tokens, tokenizer=tokenizer)

    analyze_and_finalize_circuit(circuit_model)
    

    final_results = run_evaluation(model_to_eval=circuit_model, model_name="Final Pruned Circuit (Optimal Thresholds)", full_model_for_faithfulness=full_model, dataloader=test_dataloader, device=DEVICE, two_digit_tokens=two_digit_tokens)