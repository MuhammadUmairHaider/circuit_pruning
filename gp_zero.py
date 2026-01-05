import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import time

# ==============================================================================
# IMPORTS (Switched to Zero Ablation Model)
# ==============================================================================
# Ensure this points to your zero ablation implementation
from models.gpt2_zero import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, PruningConfig
from dataset.gp import GPDataset, load_or_generate_gp_data, run_evaluation, filter_dataset_by_model_correctness
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION (Adapted for Zero Ablation)
# ==============================================================================
PRUNING_FACTOR = 10.0  # Increased to match Zero Ablation reference

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
    lambda_mlp_hidden: float = 15.0 * PRUNING_FACTOR
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 10.0 * PRUNING_FACTOR
    
    # Attention neuron pruning
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 10.0 * PRUNING_FACTOR
    
    # Embedding pruning
    prune_embedding: bool = False
    lambda_embedding: float = 1.0 * PRUNING_FACTOR
    
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
# MAIN EXECUTION FOR GENDER PRONOUNS TASK (ZERO ABLATION)
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2'
    NUM_EPOCHS = 500 # Adjust as needed
    LEARNING_RATE = 3e-2
    BATCH_SIZE = 64
    MAX_SEQ_LEN = 32
    ACCURACY_BUDGET = 0.05  # Allow 5% accuracy drop from baseline
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruning_config = PruningConfig()
    
    # --- Model and Tokenizer Setup ---
    print("\n--- Loading Models ---")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load circuit discovery model (Zero Ablation Wrapper)
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, pruning_config).to(DEVICE)
    circuit_model.eval() 

    # Load full baseline model (frozen)
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters(): 
        param.requires_grad = False

    # ----- Disable all built-in dropout layers ---
    print("\n--- Disabling all built-in dropout layers in the circuit model ---")
    disable_dropout(circuit_model)
    
    # --- Freeze the base model and unfreeze only the gates ---
    print("\n--- Configuring Trainable Parameters ---")
    total_params = 0
    trainable_params = 0
    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        if 'gate' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params += param.numel()
            
    print(f"Total parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\n--- Setting up Gender Pronouns dataset ---")
    # 1. Load Raw Data
    test_data = load_or_generate_gp_data(split="test", num_samples=100000)
    train_data = load_or_generate_gp_data(split="train", num_samples=100000)
    val_data = load_or_generate_gp_data(split="validation", num_samples=10000)

    # 2. Filter Datasets based on Base Model correctness
    print("\n--- Filtering datasets based on Base Model correctness ---")
    
    # Optional: Un-comment if you want to filter training data too
    # train_data = filter_dataset_by_model_correctness(train_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE)
    
    val_data = filter_dataset_by_model_correctness(
        val_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE
    )
    
    test_data = filter_dataset_by_model_correctness(
        test_data, full_model, tokenizer, DEVICE, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE
    )

    print(f"\nFinal Train samples: {len(train_data)}")
    print(f"Final Val samples: {len(val_data)}")
    print(f"Final Test samples: {len(test_data)}")
    
    # 3. Create Dataset Objects
    train_dataset = GPDataset(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = GPDataset(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = GPDataset(test_data, tokenizer, max_length=MAX_SEQ_LEN)
    
    # 4. Create Dataloaders
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
    
    print(f"\n--- Starting training to find 'Gender Pronouns' circuit (Zero Ablation) ---")
    print(f"Target: Maintain accuracy within {ACCURACY_BUDGET*100}% of baseline ({base_accuracy:.4f})")
    
    circuit_model.train()
    total_steps = 0
    
    # Using the cleaner tqdm loop from the reference
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_pbar:
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0
        epoch_task_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Move batch to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor): 
                    batch[key] = val.to(DEVICE)
            
            # Forward pass through circuit model
            # NOTE: For Zero Ablation, we do NOT pass corrupted_input_ids
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            # Get target outputs from full model (for KL divergence)
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask']
                )
            
            # --- KL Divergence Calculation ---
            # Calculate loss at the specific prediction position for GP task
            batch_size = circuit_outputs.logits.size(0)
            total_kl = 0
            
            for i in range(batch_size):
                pred_pos = batch['prefix_length'][i] - 1
                
                circuit_logits = circuit_outputs.logits[i, pred_pos, :]
                target_logits = target_outputs.logits[i, pred_pos, :]
                
                # KL divergence loss (Model vs Model)
                kl = F.kl_div(
                    F.log_softmax(circuit_logits, dim=-1), 
                    F.log_softmax(target_logits, dim=-1), 
                    reduction='sum', 
                    log_target=True
                )
                total_kl += kl
            
            kl_loss = total_kl / batch_size
            
            # --- Task Loss (Margin) ---
            # Extract logits for the whole batch
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

            # Margin loss
            task_loss = F.relu(1.0 - (logit_good - logit_bad)).mean()
            
            # --- Sparsity Loss ---
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            # --- Total Loss ---
            loss = task_loss*2 + sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_task_loss += task_loss.item()
            total_steps += 1
            
        # Update epoch stats
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)
        avg_task = epoch_task_loss / len(train_dataloader)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Loss': f"{avg_loss:.3f}",
            'KL': f"{avg_kl:.3f}",
            'Sprs': f"{avg_sparsity:.3f}"
        })
        
        # --- Epoch Validation ---
        # Run validation every 5 epochs or last epoch
        if ((epoch + 1) % 5 == 0) or (epoch == NUM_EPOCHS - 1):
            tqdm.write(f"\n--- Validation Epoch {epoch+1} ---")
            circuit_model.eval()
            
            val_results = run_evaluation(
                model_to_eval=circuit_model, 
                model_name=f"Circuit Epoch {epoch+1}", 
                full_model_for_faithfulness=full_model, 
                dataloader=val_dataloader, 
                device=DEVICE, 
                tokenizer=tokenizer
            )
            
            val_acc = val_results.get("accuracy", 0.0)
            accuracy_drop = base_accuracy - val_acc
            tqdm.write(f"Val Accuracy: {val_acc:.4f} (Drop: {accuracy_drop:.4f})")
            
            if accuracy_drop > ACCURACY_BUDGET:
                tqdm.write(f"WARNING: Accuracy drop exceeds budget ({ACCURACY_BUDGET})!")
            
            circuit_model.train()

    # --- Final Analysis and Pruning ---
    print("\n--- Analyzing and finalizing circuit ---")
    circuit_model.set_pruning_config(pruning_config)
    analyze_and_finalize_circuit(circuit_model)
  
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
    print("FINAL SUMMARY - Gender Pronouns Circuit Discovery (Zero Ablation)")
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
            if isinstance(value, torch.Tensor):
                 print(f"  - {key}: {value.item():.4f}")
            else:
                 print(f"  - {key}: {value:.4f}")
    print("="*60)