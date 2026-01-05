import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
import time

# --- Import your model ---
# NOTE: Keeping 'gpt2_mean' as this contains the register_mean_activations logic
from models.gpt2_mean import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, PruningConfig

# --- Import utilities ---
from dataset.ioi_t import IOIDataset, load_or_generate_ioi_data, run_evaluation
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# UPDATED PRUNING CONFIGURATION (From Latest Code)
# ==============================================================================
# PRUNING_FACTOR = 0.05

# @dataclass
# class PruningConfig:
#     init_value: float = 1.0
#     sparsity_warmup_steps: int = 0

#     # --- Fine-grained pruning ---
#     # Attention Head Pruning
#     prune_attention_heads: bool = True
#     lambda_attention_heads: float = 3 * PRUNING_FACTOR 

#     # MLP neuron pruning
#     prune_mlp_hidden: bool = True
#     lambda_mlp_hidden: float = 5 * PRUNING_FACTOR
#     prune_mlp_output: bool = True
#     lambda_mlp_output: float = 1.5 * PRUNING_FACTOR
    
#     prune_attention_neurons: bool = True
#     lambda_attention_neurons: float = 1.5 * PRUNING_FACTOR
    
#     prune_embedding: bool = False
#     lambda_embedding: float = 1 * PRUNING_FACTOR
    
#     # Prune entire attention blocks
#     prune_attention_blocks: bool = True
#     lambda_attention_blocks: float = 0.2 * PRUNING_FACTOR
    
#     # Prune entire MLP blocks
#     prune_mlp_blocks: bool = True
#     lambda_mlp_blocks: float = 0.1 * PRUNING_FACTOR
    
#     # Prune entire transformer layers
#     prune_full_layers: bool = False
#     lambda_full_layers: float = 0.000000005 * PRUNING_FACTOR


PRUNING_FACTOR = 5
@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 0

    # --- Fine-grained pruning (existing) ---
    # Attention Head Pruning
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 3 * PRUNING_FACTOR # 0.027 * PRUNING_FACTOR

    # MLP neuron pruning
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 5 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.5 * PRUNING_FACTOR
    
    
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.5 * PRUNING_FACTOR
    
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR
    
    # Prune entire attention blocks
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.2 * PRUNING_FACTOR
    
    # Prune entire MLP blocks
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.1 * PRUNING_FACTOR
    
    # Prune entire transformer layers
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.000000005 * PRUNING_FACTOR

# ==============================================================================
# FUNCTION TO RECORD MEAN ACTIVATIONS
# ==============================================================================
def record_mean_activations(model: GPT2LMHeadModel, dataloader: DataLoader, device: str) -> Dict[str, torch.Tensor]:
    """
    Runs the model over a dataset to record the mean activation of specified components.
    These mean activations will serve as the "ablated" state for each component.
    """
    model.eval()
    activations = defaultdict(list)
    hooks = []

    def get_activation_hook(name: str):
        def hook(module, input, output):
            # Capture output, handle tuple outputs (like from attention)
            activation_tensor = output[0] if isinstance(output, tuple) else output
            # Calculate mean over batch and sequence, leaving feature dimension
            activations[name].append(activation_tensor.detach().mean(dim=[0, 1]).cpu())
        return hook

    print("Attaching forward hooks to the model to record activations...")
    # Hook after embedding + positional encoding + dropout
    hooks.append(model.transformer.drop.register_forward_hook(get_activation_hook('embedding_output')))

    for i, block in enumerate(model.transformer.h):
        # Hook for Attention block output (before residual)
        hooks.append(block.attn.register_forward_hook(get_activation_hook(f'h.{i}.attn_output')))
        # Hook for MLP hidden activation (after GeLU)
        hooks.append(block.mlp.act.register_forward_hook(get_activation_hook(f'h.{i}.mlp_hidden_act')))
        # Hook for MLP block output (before residual)
        hooks.append(block.mlp.register_forward_hook(get_activation_hook(f'h.{i}.mlp_output')))
        # Hook for the output of the entire block (input to next LayerNorm)
        if i + 1 < len(model.transformer.h):
            hooks.append(model.transformer.h[i+1].ln_1.register_forward_hook(get_activation_hook(f'h.{i}.block_output')))

    # Special case for the final block's output
    hooks.append(model.transformer.ln_f.register_forward_hook(get_activation_hook(f'h.{len(model.transformer.h)-1}.block_output')))

    print(f"Recording mean activations across {len(dataloader.dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Recording Activations"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()
    print("Hooks removed.")

    # Average the activations across all batches
    mean_activations = {name: torch.stack(act_list).mean(0) for name, act_list in activations.items()}
    
    print("\nFinished recording mean activations.")
    return mean_activations

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 200
    LEARNING_RATE = 3e-3
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 64
    ACCURACY_BUDGET = 0.05
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pruning_config = PruningConfig()

    # --- Model and Tokenizer Setup ---
    print(f"Using device: {DEVICE}")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Full Model (Reference & Recording)
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    # Load Prunable Model
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE)

    print("\n--- Disabling all built-in dropout layers in the circuit model ---")
    disable_dropout(circuit_model)
    
    # --- Freeze base model and unfreeze only the gates ---
    print("\nFreezing base model weights and unfreezing gate parameters...")
    total_params = 0
    trainable_params = 0
    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        if 'gate' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"  Unfreezing for training: {name}")
        else:
            param.requires_grad = False
            
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup (Updated sizes) ---
    print("\nSetting up IOI dataset...")
    train_data = load_or_generate_ioi_data(split="train", num_samples=400)
    val_data = load_or_generate_ioi_data(split="validation", num_samples=200)
    test_data = load_or_generate_ioi_data(split="test")

    train_dataset = IOIDataset(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = IOIDataset(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = IOIDataset(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    # --- Baseline Evaluation ---
    print("\n--- Baseline evaluation on full model ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline Full Model",
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
        full_model_for_faithfulness=full_model
    )
    base_accuracy = baseline_results.get("accuracy", 0.0)
    base_logit_diff = baseline_results.get("logit_diff", 0.0)

    # --- MEAN ACTIVATION SETUP ---
    print("\n--- STEP 1: Recording Mean Activations from the full model ---")
    # We use the clean training data to get a representative sample
    mean_activations = record_mean_activations(full_model, train_dataloader, DEVICE)
    
    print("\n--- STEP 2: Registering Mean Activations with the Circuit Model ---")
    circuit_model.register_mean_activations(mean_activations)

    # --- Initial Evaluation ---
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
    
    print(f"\n--- STEP 3: Starting training with Mean Activation Patching ---")
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

            # --- MODEL CALL (Mean Ablation) ---
            # NOTE: We do NOT pass corrupted_input_ids here.
            # The model uses the registered mean activations internally for patching.
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Get target outputs from the clean run on the full model
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

            # Calculate KL divergence loss
            batch_size = circuit_outputs.logits.size(0)
            total_kl = 0
            
            for i in range(batch_size):
                t_start = batch['T_Start'][i].item() - 1
                t_end = batch['T_End'][i].item() - 1
                
                # Get valid sequence length
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, valid_length)
                
                if t_start < end_pos:
                    circuit_logits = circuit_outputs.logits[i, t_start:end_pos, :]
                    target_logits = target_outputs.logits[i, t_start:end_pos, :]
                    
                    kl = F.kl_div(
                        F.log_softmax(circuit_logits, dim=-1),
                        F.log_softmax(target_logits, dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )
                    total_kl += kl
            
            kl_loss = total_kl / batch_size
            
            # Sparsity loss
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            # Total loss
            loss = kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1
            
        # Print epoch statistics
        time_end = time.time()
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} | KL Loss: {avg_kl:.4f} | Sparsity: {avg_sparsity:.4f} | Time: {time_end - time_start:.2f}s")


    # --- Final Analysis and Pruning ---
    print("\n--- Analyzing and finalizing circuit ---")
    analyze_and_finalize_circuit(circuit_model)

    # --- Final Evaluation on Test Set ---
    print("\n--- Final evaluation on test set ---")
    circuit_model.eval()
    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Final Pruned Circuit (Mean Patching)",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer
    )

    # --- Summary ---
    print("\n" + "="*60)
    print("FINAL SUMMARY - IOI Circuit Discovery (Mean Activation Patching)")
    print("="*60)
    print(f"Baseline Accuracy: {base_accuracy:.4f}")
    print(f"Baseline Logit Diff: {base_logit_diff:.4f}")
    print(f"Final Circuit Accuracy: {final_results['accuracy']:.4f} (drop: {base_accuracy - final_results['accuracy']:.4f})")
    print(f"Final Circuit Logit Diff: {final_results['logit_diff']:.4f}")
    print(f"Final KL Divergence: {final_results['kl_div']:.4f}")
    
    # Get sparsity statistics
    sparsity_stats = circuit_model.get_sparsity_loss(step=total_steps)
    print(f"\nSparsity Statistics:")
    for key, value in sparsity_stats.items():
        if key != 'total_sparsity':
            print(f"  - {key}: {value:.4f}")
    print("="*60)