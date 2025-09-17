import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import random
from models.gpt2_test import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, GPT2LMHeadModel, PruningConfig
from dataset.ioi_t import IOIDataset, load_or_generate_ioi_data, run_evaluation

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
PRUNING_FACTOR = 0.15

# @dataclass
@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 0

    # --- Fine-grained pruning (existing) ---
    # Attention Head Pruning
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.0001 * PRUNING_FACTOR

    # MLP neuron pruning
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 0.000005 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 0.0000005 * PRUNING_FACTOR
    
    
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.0002 * PRUNING_FACTOR
    
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR
    
    # Prune entire attention blocks
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.01 * PRUNING_FACTOR
    
    # Prune entire MLP blocks
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.03 * PRUNING_FACTOR
    
    # Prune entire transformer layers
    prune_full_layers: bool = True
    lambda_full_layers: float = 0.05 * PRUNING_FACTOR

import time
# ==============================================================================
# MAIN EXECUTION FOR IOI TASK
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 200
    LEARNING_RATE = 5e-3
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 64
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
            param.requires_grad = False
        else:
            print(f"  Unfreezing for training: {name}")
            param.requires_grad = True
            trainable_params += param.numel()
            
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    print("\nVerifying trainable parameters:")
    for name, param in circuit_model.named_parameters():
        if param.requires_grad:
            print(f"  TRAINABLE: {name} - shape: {param.shape}")

    # # Double-check optimizer
    # print(f"\nOptimizer is training {len(optimizer.param_groups[0]['params'])} parameter tensors")

    # --- Dataset Setup ---
    print("\nSetting up IOI dataset...")
    # Load from disk
    train_data = load_or_generate_ioi_data(split="train_100k", num_samples=1000)  # Limit samples for efficiency
    val_data = load_or_generate_ioi_data(split="validation", num_samples=1000)
    test_data = load_or_generate_ioi_data(split="test")

    # Create dataset objects
    train_dataset = IOIDataset(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = IOIDataset(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = IOIDataset(test_data, tokenizer, max_length=MAX_SEQ_LEN)

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
        dataloader=val_dataloader, 
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
    initial_accuracy = initial_results.get("accuracy", 0.0)
    initial_logit_diff = initial_results.get("logit_diff", 0.0)

    # --- Training ---
    # The optimizer will now only see the parameters that require gradients (the gates)
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    
    
    print(f"\n--- Starting training to find 'Indirect Object Identification' circuit ---")
    print(f"Target: Maintain accuracy within {ACCURACY_BUDGET*100}% of baseline ({base_accuracy:.4f})")

    circuit_model.train()
    total_steps = 0
    time_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            
            # Move batch to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)
            
            # Forward pass through circuit model with corrupted inputs
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
                # Fix 1: Use T_Start instead of Start
                t_start = batch['T_Start'][i].item()-1
                t_end = batch['T_End'][i].item()-1
                
                # if(len(batch['target_tokens'][i])>1):
                #     print(len(batch['target_tokens'][i]), batch['target_tokens'][i], tokenizer.decode(batch['target_tokens'][i]))
                
                # Fix 2: Get valid sequence length to avoid padding
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, valid_length)
                
                # Fix 3: Ensure we don't go out of bounds
                if t_start < end_pos:
                    circuit_logits = circuit_outputs.logits[i, t_start:end_pos, :]
                    target_logits = target_outputs.logits[i, t_start:end_pos, :]
                    
                    # KL divergence loss
                    kl = F.kl_div(
                        F.log_softmax(circuit_logits, dim=-1),
                        F.log_softmax(target_logits, dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )
                    total_kl += kl
            
            kl_loss = total_kl / batch_size
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            # Total loss
            loss = kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1
        time_end = time.time()
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} | KL Loss: {epoch_kl_loss:.4f} | Sparsity Loss: {epoch_sparsity_loss:.4f} | Time: {time_end - time_start:.2f}s")
        # Print epoch statistics
        # avg_loss = epoch_loss / len(train_dataloader)
        # avg_kl = epoch_kl_loss / len(train_dataloader)
        # avg_sparsity = epoch_sparsity_loss / len(train_dataloader)
        
        # print(f"\nEpoch {epoch+1} Summary:")
        # print(f"  - Total Loss: {avg_loss:.4f}")
        # print(f"  - KL Loss: {avg_kl:.4f}")
        # print(f"  - Sparsity Loss: {avg_sparsity:.4f}")
        
        # --- Epoch Validation ---
        # circuit_model.eval()
        # val_results = run_evaluation(
        #     model_to_eval=circuit_model, 
        #     model_name=f"Circuit after Epoch {epoch+1}", 
        #     full_model_for_faithfulness=full_model, 
        #     dataloader=val_dataloader, 
        #     device=DEVICE, 
        #     tokenizer=tokenizer
        # )
        
        # Check if we're within accuracy budget
        # current_accuracy = val_results.get("accuracy", 0.0)
        # accuracy_drop = base_accuracy - current_accuracy
        # if accuracy_drop > ACCURACY_BUDGET:
        #     print(f"  WARNING: Accuracy drop ({accuracy_drop:.4f}) exceeds budget ({ACCURACY_BUDGET})!")
        
        circuit_model.train()
    

    # --- Final Analysis and Pruning ---
    print("\n--- Analyzing and finalizing circuit ---")
    analyze_and_finalize_circuit(circuit_model)
    
    # --- Final Evaluation on Test Set ---
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
    print("FINAL SUMMARY - IOI Circuit Discovery")
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
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # Copy your HardConcreteGate class here
# class HardConcreteGate(nn.Module):
#     """
#     A gate that uses the Hard Concrete distribution to learn binary decisions.
#     """
    
#     def __init__(
#         self, 
#         size: int, 
#         beta: float = 2.0/3.0,
#         gamma: float = -0.1,
#         zeta: float = 1.1,
#         init_min: float = 0.1, 
#         init_max: float = 1.1
#     ):
#         super().__init__()
        
#         # Register buffers for distribution parameters
#         self.register_buffer("beta", torch.tensor(beta))
#         self.register_buffer("gamma", torch.tensor(gamma))
#         self.register_buffer("zeta", torch.tensor(zeta))
        
#         # Flag for final hard pruning mode
#         self.final_mode = False
        
#         # Learnable parameters
#         self.log_alpha = nn.Parameter(torch.Tensor(size))
        
#         # Initialize
#         self.init_weights(init_min, init_max)
        
#     def init_weights(self, init_min: float, init_max: float):
#         """Initialize log_alpha parameters uniformly."""
#         with torch.no_grad():
#             self.log_alpha.uniform_(init_min, init_max)
    
#     def forward(self) -> torch.Tensor:
#         """
#         Samples from the Hard Concrete distribution.
#         """
#         if self.final_mode:
#             # Hard binary decisions for final circuit
#             s = torch.sigmoid(self.log_alpha)
#             s_stretched = s * (self.zeta - self.gamma) + self.gamma
#             gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
#             return (gate > 0.5).float()
        
#         if self.training:
#             # Sample from Hard Concrete during training
#             u = torch.rand_like(self.log_alpha)
#             u = u.clamp(1e-8, 1.0 - 1e-8)  # Numerical stability
            
#             # Reparameterization trick with Gumbel-Softmax
#             s = torch.sigmoid(
#                 (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
#             )
#         else:
#             # Expected value during evaluation
#             s = torch.sigmoid(self.log_alpha)
        
#         # Stretch and clip
#         s_stretched = s * (self.zeta - self.gamma) + self.gamma
#         gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
        
#         return gate
    
#     def set_final_mode(self, mode: bool = True):
#         """Enable/disable final hard pruning mode."""
#         self.final_mode = mode

# def test_hardconcrete_gate_values():
#     """
#     Test HardConcreteGate to see actual values in different modes
#     """
#     print("="*80)
#     print("TESTING HARDCONCRETE GATE VALUES")
#     print("="*80)
    
#     # Create a test gate with various log_alpha values
#     gate = HardConcreteGate(size=10)
    
#     # Set some specific log_alpha values to test different scenarios
#     with torch.no_grad():
#         gate.log_alpha.data = torch.tensor([
#             -10.0,  # Very negative (should be OFF)
#             -5.0,   # Negative (should be OFF)
#             -2.0,   # Slightly negative
#             -0.5,   # Small negative
#             0.0,    # Zero
#             0.5,    # Small positive
#             2.0,    # Positive
#             5.0,    # Large positive (should be ON)
#             10.0,   # Very positive (should be ON)
#             0.1     # Just above zero
#         ])
    
#     print(f"Test log_alpha values: {gate.log_alpha.data}")
#     print(f"Gate parameters: beta={gate.beta}, gamma={gate.gamma}, zeta={gate.zeta}")
    
#     # Test 1: Training mode (should be stochastic)
#     print(f"\n{'='*60}")
#     print("TEST 1: TRAINING MODE (stochastic)")
#     print("="*60)
#     gate.train()
#     gate.set_final_mode(False)
    
#     print("Multiple samples (should vary):")
#     for i in range(5):
#         values = gate()
#         print(f"  Sample {i+1}: {values.detach().numpy()}")
    
#     # Test 2: Eval mode (should be deterministic, but continuous)
#     print(f"\n{'='*60}")
#     print("TEST 2: EVAL MODE (deterministic, continuous)")
#     print("="*60)
#     gate.eval()
#     gate.set_final_mode(False)
    
#     print("Multiple forward passes (should be identical):")
#     for i in range(3):
#         values = gate()
#         print(f"  Pass {i+1}: {values.detach().numpy()}")
    
#     # Show intermediate calculations
#     print(f"\nIntermediate calculations:")
#     with torch.no_grad():
#         s = torch.sigmoid(gate.log_alpha)
#         s_stretched = s * (gate.zeta - gate.gamma) + gate.gamma
#         gate_vals = F.hardtanh(s_stretched, min_val=0, max_val=1)
        
#         print(f"  log_alpha:   {gate.log_alpha.data.numpy()}")
#         print(f"  sigmoid(α):  {s.numpy()}")
#         print(f"  stretched:   {s_stretched.numpy()}")
#         print(f"  hardtanh:    {gate_vals.numpy()}")
    
#     # Test 3: Final mode (should be exactly 0 or 1)
#     print(f"\n{'='*60}")
#     print("TEST 3: FINAL MODE (should be exactly 0.0 or 1.0)")
#     print("="*60)
#     gate.eval()
#     gate.set_final_mode(True)
    
#     print("Multiple forward passes (should be identical and binary):")
#     for i in range(3):
#         values = gate()
#         print(f"  Pass {i+1}: {values.detach().numpy()}")
        
#         # Check if values are exactly 0 or 1
#         is_binary = torch.all((values == 0.0) | (values == 1.0))
#         print(f"    All binary (0 or 1): {is_binary}")
#         print(f"    Unique values: {torch.unique(values).numpy()}")
    
#     # Show final mode intermediate calculations
#     print(f"\nFinal mode intermediate calculations:")
#     with torch.no_grad():
#         s = torch.sigmoid(gate.log_alpha)
#         s_stretched = s * (gate.zeta - gate.gamma) + gate.gamma
#         gate_vals = F.hardtanh(s_stretched, min_val=0, max_val=1)
#         binary_vals = (gate_vals > 0.5).float()
        
#         print(f"  log_alpha:     {gate.log_alpha.data.numpy()}")
#         print(f"  sigmoid(α):    {s.numpy()}")
#         print(f"  stretched:     {s_stretched.numpy()}")
#         print(f"  hardtanh:      {gate_vals.numpy()}")
#         print(f"  > 0.5:         {(gate_vals > 0.5).numpy()}")
#         print(f"  final binary:  {binary_vals.numpy()}")
    
#     # Test 4: Edge cases
#     print(f"\n{'='*60}")
#     print("TEST 4: EDGE CASES")
#     print("="*60)
    
#     # Test with extreme values
#     extreme_gate = HardConcreteGate(size=6)
#     with torch.no_grad():
#         extreme_gate.log_alpha.data = torch.tensor([
#             -100.0,  # Extremely negative
#             -1e6,    # Machine negative
#             0.0,     # Exactly zero
#             1e6,     # Machine positive  
#             100.0,   # Extremely positive
#             0.5      # Right at threshold area
#         ])
    
#     extreme_gate.eval()
#     extreme_gate.set_final_mode(True)
    
#     print("Extreme log_alpha values test:")
#     values = extreme_gate()
#     print(f"  log_alpha: {extreme_gate.log_alpha.data.numpy()}")
#     print(f"  output:    {values.detach().numpy()}")
#     print(f"  all binary: {torch.all((values == 0.0) | (values == 1.0))}")
    
#     return gate

# def test_gate_threshold_boundary():
#     """
#     Test the exact threshold where gates switch from 0 to 1
#     """
#     print(f"\n{'='*80}")
#     print("TESTING GATE THRESHOLD BOUNDARY")
#     print("="*80)
    
#     # Test around the boundary where gates switch
#     test_alphas = torch.linspace(-3, 3, 21)  # From -3 to 3 in 21 steps
    
#     gate = HardConcreteGate(size=len(test_alphas))
#     with torch.no_grad():
#         gate.log_alpha.data = test_alphas
    
#     gate.eval()
    
#     print("Testing threshold boundary:")
#     print("log_alpha  | sigmoid  | stretched | hardtanh | >0.5 | final")
#     print("-" * 65)
    
#     # Without final mode
#     gate.set_final_mode(False)
#     soft_values = gate()
    
#     # With final mode  
#     gate.set_final_mode(True)
#     hard_values = gate()
    
#     with torch.no_grad():
#         s = torch.sigmoid(gate.log_alpha)
#         s_stretched = s * (gate.zeta - gate.gamma) + gate.gamma
#         gate_vals = F.hardtanh(s_stretched, min_val=0, max_val=1)
        
#         for i, alpha in enumerate(test_alphas):
#             print(f"{alpha:8.2f} | {s[i]:6.4f} | {s_stretched[i]:7.4f} | "
#                   f"{gate_vals[i]:6.4f} | {gate_vals[i] > 0.5:4} | {hard_values[i]:4.1f}")

# def test_model_gates(model):
#     """
#     Test actual gates in your model to see their values
#     """
#     print(f"\n{'='*80}")
#     print("TESTING ACTUAL MODEL GATES")
#     print("="*80)
    
#     model.eval()
    
#     # Test different modes
#     modes = [
#         ("Normal eval", False),
#         ("Final mode", True)
#     ]
    
#     for mode_name, final_mode in modes:
#         print(f"\n🔧 {mode_name.upper()}:")
        
#         # Set final mode for all gates
#         if hasattr(model, 'set_final_circuit_mode'):
#             model.set_final_circuit_mode(final_mode)
        
#         gate_samples = []
#         gate_names = []
        
#         with torch.no_grad():
#             # Collect samples from first few gates
#             count = 0
#             for name, module in model.named_modules():
#                 if hasattr(module, 'gate') and hasattr(module.gate, 'forward'):
#                     if count < 5:  # Just first 5 gates for testing
#                         values = module.gate()
#                         gate_samples.append(values)
#                         gate_names.append(name)
                        
#                         print(f"  {name}:")
#                         if values.numel() <= 10:  # Show all if ≤10 elements
#                             print(f"    Values: {values.cpu().numpy()}")
#                         else:  # Show first 10 if more
#                             print(f"    Values (first 10): {values.cpu().numpy()[:10]}")
                        
#                         # Check if binary
#                         is_binary = torch.all((values == 0.0) | (values == 1.0))
#                         unique_vals = torch.unique(values)
#                         print(f"    Binary: {is_binary}, Unique: {unique_vals.cpu().numpy()}")
#                         print(f"    Shape: {values.shape}, Active: {(values > 0.5).sum().item()}/{values.numel()}")
                        
#                         count += 1
                        
#                     if count >= 5:
#                         break
        
#         if len(gate_samples) == 0:
#             print("  ❌ No gates found! Check your model structure.")

# # Main test function
# if __name__ == "__main__":
#     # Test standalone HardConcreteGate
#     gate = test_hardconcrete_gate_values()
    
#     # Test threshold boundary
#     test_gate_threshold_boundary()
    
#     print(f"\n{'='*80}")
#     print("✅ TESTING COMPLETE")
#     print("="*80)
#     print("Key things to check:")
#     print("1. Final mode should produce EXACTLY 0.0 or 1.0")
#     print("2. Values should be deterministic in final mode")
#     print("3. Threshold should be around log_alpha ≈ 0.7-1.0")
#     print("4. Very negative log_alpha → 0.0")
#     print("5. Very positive log_alpha → 1.0")

# # Usage with your model:
# # Add this to your code after training:
# print("\\n=== TESTING HARDCONCRETE GATES ===")
# test_gate_values = test_hardconcrete_gate_values()
# test_gate_threshold_boundary()
# test_model_gates(circuit_model)