import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from dataclasses import dataclass

# --- Import your custom modules here ---
# Ensure these match your actual file structure
from models.gpt2_zero import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, PruningConfig
from dataset.ioi_t import IOIDataset, load_or_generate_ioi_data, run_evaluation, filter_dataset_by_model_correctness
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION
# ==============================================================================
PRUNING_FACTOR = 5.0

@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 50

    # --- Fine-grained pruning ---
    # Attention Head Pruning
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 2.0 * PRUNING_FACTOR

    # MLP neuron pruning
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 15 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 10 * PRUNING_FACTOR
    
    # Attention neuron pruning
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 10 * PRUNING_FACTOR
    
    # Embedding pruning
    prune_embedding: bool = False
    lambda_embedding: float = 1 * PRUNING_FACTOR
    
    # --- Block-level pruning ---
    # Prune entire attention blocks
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0 * PRUNING_FACTOR
    
    # Prune entire MLP blocks
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 1.0 * PRUNING_FACTOR
    
    # Prune entire transformer layers
    prune_full_layers: bool = True
    lambda_full_layers: float = 0.000000005 * PRUNING_FACTOR


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
    ACCURACY_BUDGET = 0.05  # Allow 5% accuracy drop from baseline
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruning_config = PruningConfig()
    
    # --- Model and Tokenizer Setup ---
    print("\n--- Loading Models ---")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load circuit discovery model (wrapper)
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE)
    circuit_model.eval() # Start in eval mode for initial checks

    # Load full baseline model (frozen)
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters(): 
        param.requires_grad = False

    # ----- Disable all built-in dropout layers in the circuit model ---
    print("\n--- Disabling all built-in dropout layers in the circuit model ---")
    disable_dropout(circuit_model)
    # -----------------------------------------------------------------
    
    # --- Freeze the base model and unfreeze only the gates ---
    print("\n--- Configuring Trainable Parameters ---")
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
            
    print(f"Total parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\n--- Setting up IOI dataset ---")
    
    # 1. Load Raw Data
    test_data = load_or_generate_ioi_data(split="test", num_samples=1000) 
    train_data = load_or_generate_ioi_data(split="train", num_samples=200)
    val_data = load_or_generate_ioi_data(split="validation", num_samples=200)

    # 2. Filter Datasets based on Base Model correctness
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
    
    # Print some dataset examples
    print("\nSome examples from the training dataset:")
    for i in range(2):
        sample = train_dataset[i]
        input_ids = sample['input_ids']
        target_tokens = sample['target_tokens']
        
        print(f"\nExample {i+1}:")
        print("Input Text: ", tokenizer.decode(input_ids, skip_special_tokens=True))
        print("Target Tokens: ", tokenizer.decode(target_tokens, skip_special_tokens=True))
    
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
    initial_results = run_evaluation(
        model_to_eval=circuit_model, 
        model_name="Initial Circuit Model", 
        full_model_for_faithfulness=full_model, 
        dataloader=val_dataloader, 
        device=DEVICE, 
        tokenizer=tokenizer
    )
    initial_accuracy = initial_results.get("accuracy", 0.0)

    # --- Training ---
    # --- Training ---
    # The optimizer will now only see the parameters that require gradients (the gates)
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    print(f"\n--- Starting training to find 'Indirect Object Identification' circuit ---")
    print(f"Target: Maintain accuracy within {ACCURACY_BUDGET*100}% of baseline ({base_accuracy:.4f})")

    circuit_model.train()
    total_steps = 0
    
    # 1. Move tqdm to the OUTER loop
    # We use dynamic_ncols to adjust to terminal width automatically
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0
        epoch_task_loss = 0
        
        # 2. Iterate dataloader normally (no tqdm here)
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Move batch to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)
            
            # Forward pass
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            
            # --- KL Divergence Loss ---
            batch_size = circuit_outputs.logits.size(0)
            total_kl = 0
            
            for i in range(batch_size):
                t_start = batch['T_Start'][i].item() - 1 
                t_end = batch['T_End'][i].item() - 1
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, valid_length)
                
                if t_start < end_pos:
                    circuit_logits = circuit_outputs.logits[i, t_start:end_pos]
                    target_logits = target_outputs.logits[i, t_start:end_pos]
                    
                    kl = F.kl_div(
                        F.log_softmax(circuit_logits, dim=-1),
                        F.log_softmax(target_logits, dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )
                    total_kl += kl
            
            kl_loss = total_kl / batch_size

            # --- Task Loss (Margin) ---
            pos_good = batch['T_Start'] - 1 
            token_good = batch['target_tokens'][:, 0]
            token_bad = batch['distractor_tokens'][:, 0]
            batch_indices = torch.arange(batch_size, device=DEVICE)
            
            logit_good = circuit_outputs.logits[batch_indices, pos_good, token_good]
            logit_bad = circuit_outputs.logits[batch_indices, pos_good, token_bad]

            task_loss = F.relu(2.0 - (logit_good - logit_bad)).mean()
            
            # --- Sparsity Loss ---
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            # --- Total Loss ---
            loss = kl_loss + sparsity_loss + task_loss
            
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_task_loss += task_loss.item()
            total_steps += 1
        
        # End of Epoch Stats
        epoch_loss /= len(train_dataloader)
        epoch_kl_loss /= len(train_dataloader)
        epoch_sparsity_loss /= len(train_dataloader)
        epoch_task_loss /= len(train_dataloader)

        # 3. Update the progress bar status (This replaces the print spam)
        # Tqdm handles time automatically, so we just add the metrics
        epoch_pbar.set_postfix({
            'Loss': f"{epoch_loss:.3f}",
            'KL': f"{epoch_kl_loss:.3f}",
            'Task': f"{epoch_task_loss:.3f}",
            'Sprs': f"{epoch_sparsity_loss:.3f}"
        })
        
        # Validation
        if ((epoch + 1) % 10 == 0) or (epoch == NUM_EPOCHS - 1):
            # 4. Use tqdm.write so validation logs appear cleanly ABOVE the bar
            tqdm.write(f"\n--- Validation Epoch {epoch+1} ---")
            
            circuit_model.eval()
            # Suppress internal prints of run_evaluation if possible, or accept they might print
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Validation Epoch {epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_dataloader,
                device=DEVICE,
                tokenizer=tokenizer
            )
            # Log the key validation metric using tqdm.write
            val_acc = val_results.get("accuracy", 0.0)
            tqdm.write(f"Validation Accuracy: {val_acc:.4f}")
            
            circuit_model.train()
    
    # --- Final Analysis and Pruning ---
    print("\n--- Analyzing and finalizing circuit ---")
    analyze_and_finalize_circuit(circuit_model)
    
    # --- Final Evaluation on Test Set ---
    print("\n--- Final evaluation on test set ---")
    circuit_model.eval()
    final_results = run_evaluation(
        model_to_eval=circuit_model, 
        model_name="Final Pruned Circuit (Zero Ablation)", 
        full_model_for_faithfulness=full_model, 
        dataloader=test_dataloader, 
        device=DEVICE, 
        tokenizer=tokenizer
    )
    
    # --- Summary ---
    print("\n" + "="*60)
    print("FINAL SUMMARY - IOI Circuit Discovery (Zero Ablation)")
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
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.item():.4f}")
            else:
                print(f"  - {key}: {value:.4f}")
    print("="*60)