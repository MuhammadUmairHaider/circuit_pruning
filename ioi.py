import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assume your models and helper functions are in these locations
from models.gpt2_circuit import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2, PruningConfig
from utils import analyze_and_finalize_circuit, disable_dropout

# Import the NEW data preparation pipeline and the PyTorch Dataset class
from dataset.ioi_task import prepare_ioi_dataset, IOIDataset, run_ioi_evaluation, run_corrupted_only_test, test_pure_token_bias, analyze_dataset_balance, test_swapped_roles

from dataclasses import dataclass

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

prune_factor = 1.0
@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 0

    # --- Control Panel for Pruning Granularity ---
    
    # Attention Head Pruning (what we already have)
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.01 * prune_factor  # The penalty for the attention head gates

    # --- NEW: Separate controls for each MLP layer ---
    prune_mlp_hidden: bool = True       # Prune the intermediate "fat" layer of the MLP
    lambda_mlp_hidden: float = 0.0005 * prune_factor     # The penalty for the hidden layer gates

    prune_mlp_output: bool = True      # Prune the final output of the entire MLP sub-block
    lambda_mlp_output: float = 0.0005 * prune_factor    # The penalty for the output gates
    
    prune_embedding: bool = False
    lambda_embedding: float = 0.1 * prune_factor# This is a crucial hyperparameter to tune
    


if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'gpt2-xl'
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 8
    MAX_SEQ_LEN = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # pruning_config = PruningConfig()
    
    # --- Tokenizer and Model Setup ---
    print(f"Loading tokenizer and models for {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(MODEL_NAME, PruningConfig).to(DEVICE)
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False
    
    disable_dropout(circuit_model)
    
    # --- Freeze non-gate parameters ---
    print("Freezing base model weights and unfreezing gate parameters...")
    for name, param in circuit_model.named_parameters():
        if 'gate' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print("Loading dataset from disk...")
    DATASET_PATH = "/u/amo-d1/grad/mha361/work/circuits/data/edge_pruning/datasets/ioi" # Your path here

    # 1. Load the raw HuggingFace datasets from disk
    train_data_raw = prepare_ioi_dataset(DATASET_PATH, split='train')
    val_data_raw = prepare_ioi_dataset(DATASET_PATH, split='validation') 
    test_data_raw = prepare_ioi_dataset(DATASET_PATH, split='test')

    # 2. Create PyTorch Datasets and DataLoaders from the loaded data
    train_dataset = IOIDataset(train_data_raw, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = IOIDataset(val_data_raw, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = IOIDataset(test_data_raw, tokenizer, max_length=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    # --- End of new code ---
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Baseline & Initial Evaluation ---
    run_ioi_evaluation(full_model, "Baseline Full Model", val_dataloader, device=DEVICE, full_model_for_faithfulness=full_model)
    run_ioi_evaluation(circuit_model, "Initial Circuit Model", val_dataloader, device=DEVICE, full_model_for_faithfulness=full_model)
    
    run_corrupted_only_test(full_model, "Baseline Full Model", val_dataloader, device=DEVICE)

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)
    
    print(f"\n--- Starting training to find IOI circuit ---")
    circuit_model.train()
    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor): batch[key] = val.to(DEVICE)
            
            # Forward pass with interchange ablation
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

            last_token_circuit_logits = circuit_outputs.logits.gather(1, batch['last_token_idx'].view(-1, 1, 1).expand(-1, -1, circuit_outputs.logits.size(-1))).squeeze(1)
            last_token_target_logits = target_outputs.logits.gather(1, batch['last_token_idx'].view(-1, 1, 1).expand(-1, -1, target_outputs.logits.size(-1))).squeeze(1)

            # Faithfulness (KL) + Sparsity Loss
            kl_loss = F.kl_div(F.log_softmax(last_token_circuit_logits, dim=-1), F.log_softmax(last_token_target_logits, dim=-1), reduction='batchmean', log_target=True)
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            loss = kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            total_steps += 1
        
        # --- Epoch Validation ---
        run_ioi_evaluation(circuit_model, f"Circuit after Epoch {epoch+1}", val_dataloader, device=DEVICE, full_model_for_faithfulness=full_model)
        circuit_model.train()

    # --- Final Analysis ---
    print("\n--- Training finished. Analyzing final circuit. ---")
    analyze_and_finalize_circuit(circuit_model)
    run_ioi_evaluation(circuit_model, "Final Pruned Circuit", test_dataloader, device=DEVICE, full_model_for_faithfulness=full_model )