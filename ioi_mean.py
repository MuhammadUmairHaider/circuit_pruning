import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from tqdm import tqdm
from collections import defaultdict

# --- Import your new model and config ---
# Make sure the filename matches what you saved
from models.gpt2_mean import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, PruningConfig

# --- Import your project's utility and dataset files ---
# Ensure these paths are correct for your project structure
from dataset.ioi_t import IOIDataset, load_or_generate_ioi_data, run_evaluation
from utils import disable_dropout, analyze_and_finalize_circuit
from models.l0 import HardConcreteGate



from dataclasses import dataclass
PRUNING_FACTOR = 0.4

# @dataclass
@dataclass
class PruningConfig:
    init_value: float = 1.0
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
    lambda_attention_blocks: float = 0.01 * PRUNING_FACTOR
    
    # Prune entire MLP blocks
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.03 * PRUNING_FACTOR
    
    # Prune entire transformer layers
    prune_full_layers: bool = True
    lambda_full_layers: float = 0.05 * PRUNING_FACTOR

# ==============================================================================
# NEW: FUNCTION TO RECORD MEAN ACTIVATIONS
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
            # The hook function is called after a module's forward pass.
            # We capture the output tensor.
            activation_tensor = output[0] if isinstance(output, tuple) else output
            # We calculate the mean over the batch and sequence dimensions,
            # leaving only the feature dimension. This gives us a single "average" vector.
            activations[name].append(activation_tensor.detach().mean(dim=[0, 1]).cpu())
        return hook

    print("Attaching forward hooks to the model to record activations...")
    # Hook after embedding + positional encoding + dropout
    hooks.append(model.transformer.drop.register_forward_hook(get_activation_hook('embedding_output')))

    for i, block in enumerate(model.transformer.h):
        # Hook for Attention block output (before the residual connection is added)
        hooks.append(block.attn.register_forward_hook(get_activation_hook(f'h.{i}.attn_output')))
        # Hook for MLP hidden activation (after the GeLU/activation function)
        hooks.append(block.mlp.act.register_forward_hook(get_activation_hook(f'h.{i}.mlp_hidden_act')))
        # Hook for MLP block output (before the residual connection is added)
        hooks.append(block.mlp.register_forward_hook(get_activation_hook(f'h.{i}.mlp_output')))
        # Hook for the output of the entire block. We capture this by hooking the input
        # to the *next* block's first LayerNorm.
        if i + 1 < len(model.transformer.h):
            hooks.append(model.transformer.h[i+1].ln_1.register_forward_hook(get_activation_hook(f'h.{i}.block_output')))

    # Special case for the final block's output (input to the final LayerNorm)
    hooks.append(model.transformer.ln_f.register_forward_hook(get_activation_hook(f'h.{len(model.transformer.h)-1}.block_output')))

    print(f"Recording mean activations across {len(dataloader.dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Recording Activations"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    # Clean up by removing all hooks
    for hook in hooks:
        hook.remove()
    print("Hooks removed.")

    # Average the activations across all batches to get the final mean vectors
    mean_activations = {name: torch.stack(act_list).mean(0) for name, act_list in activations.items()}
    
    print("\nFinished recording mean activations for the following components:")
    for name, tensor in mean_activations.items():
        print(f"  - {name}: shape {tensor.shape}")
        
    return mean_activations

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 200
    LEARNING_RATE = 5e-3
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

    # Load the unaltered model for generating target logits and for recording activations
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    # Load your new prunable model
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE)

    print("\n--- Disabling all built-in dropout layers in the circuit model ---")
    disable_dropout(circuit_model)
    
    # --- Freeze base model and unfreeze only the gates ---
    print("\nFreezing base model weights and unfreezing gate parameters...")
    for name, param in circuit_model.named_parameters():
        if 'gate' in name:
            param.requires_grad = True
            print(f"  Unfreezing for training: {name}")
        else:
            param.requires_grad = False

    # --- Dataset Setup ---
    print("\nSetting up IOI dataset...")
    train_data = load_or_generate_ioi_data(split="train_100k", num_samples=2000)
    val_data = load_or_generate_ioi_data(split="validation", num_samples=1000)
    test_data = load_or_generate_ioi_data(split="test", num_samples=1000)

    train_dataset = IOIDataset(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = IOIDataset(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = IOIDataset(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Baseline Evaluation ---
    print("\n--- Baseline evaluation on full model ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline Full Model",
        dataloader=val_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
        full_model_for_faithfulness=full_model
    )
    base_accuracy = baseline_results.get("accuracy", 0.0)

    # --- NEW: Record and Register Mean Activations ---
    print("\n--- STEP 1: Recording Mean Activations from the full model ---")
    # We use the clean training data to get a representative sample of activations
    mean_activations = record_mean_activations(full_model, train_dataloader, DEVICE)
    
    print("\n--- STEP 2: Registering Mean Activations with the Circuit Model ---")
    circuit_model.register_mean_activations(mean_activations)

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    print(f"\n--- STEP 3: Starting training with Mean Activation Patching ---")
    circuit_model.train()
    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss, epoch_kl_loss, epoch_sparsity_loss = 0, 0, 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            # --- MODIFIED MODEL CALL ---
            # We no longer provide `corrupted_input_ids`. The model will use the
            # registered mean activations internally for its patching logic.
            circuit_outputs = circuit_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get target outputs from the clean run on the full model
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Calculate KL divergence loss (same as before)
            batch_size = circuit_outputs.logits.size(0)
            total_kl = 0
            for i in range(batch_size):
                Start = batch['T_Start'][i] - 1
                End = batch['T_End'][i] - 1
                circuit_logits = circuit_outputs.logits[i, Start:End, :]
                target_logits = target_outputs.logits[i, Start:End, :]
                kl = F.kl_div(
                    F.log_softmax(circuit_logits, dim=-1),
                    F.log_softmax(target_logits, dim=-1),
                    reduction='sum',
                    log_target=True
                )
                total_kl += kl
            kl_loss = total_kl / batch_size
            
            # Sparsity loss
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            # Total loss
            # kl_loss = kl_loss * 2
            loss = kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1
            
        # Print epoch statistics
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  - Total Loss: {avg_loss:.4f}")
        print(f"  - KL Loss (Task Performance): {avg_kl:.4f}")
        print(f"  - Sparsity Loss (Regularization): {avg_sparsity:.4f}")
        
        # Optional: Run validation every epoch (can be slow)
        # circuit_model.eval()
        # run_evaluation(...)
        # circuit_model.train()

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
    # ... your summary printing logic ...
    print(f"Baseline Accuracy: {base_accuracy:.4f}")
    print(f"Final Circuit Accuracy: {final_results['accuracy']:.4f}")
    print(f"Final Circuit Logit Diff: {final_results['logit_diff']:.4f}")
    print("="*60)