import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import wandb
from tqdm import tqdm
from dataclasses import dataclass, asdict

# Ensure we can import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.gpt2_circuit import PrunableGPT2LMHeadModel as CircuitDiscoveryGPT2
from dataset.ioi import IOIDataset, load_or_generate_ioi_data, run_evaluation, filter_dataset_by_model_correctness
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# CONFIGURATION 
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    # Environment
    model_name: str = 'gpt2'
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir: str = './outputs/ioi_pruning'
    
    # Training
    num_epochs: int = 500
    learning_rate: float = 3e-2
    batch_size: int = 32
    max_seq_len: int = 32
    accuracy_budget: float = 0.05
    
    # Pruning Config equivalent values
    pruning_factor: float = 1.0
    init_value: float = 1.0
    sparsity_warmup_steps: int = 500
    depth_penalty_scaling: float = 0.0
    
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 1.0
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0
    prune_embedding: bool = False
    lambda_embedding: float = 1.0
    
    # Automated Loop behavior
    use_wandb: bool = True
    wandb_project: str = "Circuit-Pruning"
    eval_every: int = 50
    save_best: bool = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Automated & Stable IOI Pruning process.")
    # Add minimal args to override Config, rest fall back to dataclass defaults
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-2, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every N epochs")
    parser.add_argument("--use_wandb", type=str2bool, nargs='?', const=True, default=True, help="Use W&B")
    parser.add_argument("--output_dir", type=str, default="./outputs/ioi_pruning", help="Output directory paths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    return args

def build_config(args):
    config = Config(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        use_wandb=args.use_wandb,
        output_dir=args.output_dir,
        seed=args.seed
    )
    return config

# Dynamic recreation of PruningConfig exactly as required by models.gpt2_circuit
@dataclass
class PruningConfig:
    init_value: float = 1.0
    sparsity_warmup_steps: int = 500
    depth_penalty_scaling: float = 0.0
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0 
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 1.0
    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0
    prune_embedding: bool = False
    lambda_embedding: float = 1.0

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)
    
    print(f"\n--- Staring Automated IOI Pruning ---")
    print(f"Output Directory: {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    if cfg.use_wandb:
        print(f"Initializing W&B in project '{cfg.wandb_project}'...")
        # Offline model ensures batch scripts won't block forever if connection drops
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))
        wandb.run.name = f"ioi_automated_pruning_{wandb.run.id}"
    else:
        print("W&B logging is disabled.")

    pruning_config = PruningConfig(
        init_value=cfg.init_value,
        sparsity_warmup_steps=cfg.sparsity_warmup_steps,
        depth_penalty_scaling=cfg.depth_penalty_scaling,
        prune_attention_heads=cfg.prune_attention_heads,
        lambda_attention_heads=cfg.lambda_attention_heads * cfg.pruning_factor,
        prune_mlp_hidden=cfg.prune_mlp_hidden,
        lambda_mlp_hidden=cfg.lambda_mlp_hidden * cfg.pruning_factor,
        prune_mlp_output=cfg.prune_mlp_output,
        lambda_mlp_output=cfg.lambda_mlp_output * cfg.pruning_factor,
        prune_attention_neurons=cfg.prune_attention_neurons,
        lambda_attention_neurons=cfg.lambda_attention_neurons * cfg.pruning_factor,
        prune_attention_blocks=cfg.prune_attention_blocks,
        lambda_attention_blocks=cfg.lambda_attention_blocks * cfg.pruning_factor,
        prune_mlp_blocks=cfg.prune_mlp_blocks,
        lambda_mlp_blocks=cfg.lambda_mlp_blocks * cfg.pruning_factor,
        prune_full_layers=cfg.prune_full_layers,
        lambda_full_layers=cfg.lambda_full_layers * cfg.pruning_factor,
        prune_embedding=cfg.prune_embedding,
        lambda_embedding=cfg.lambda_embedding * cfg.pruning_factor
    )

    # --- Model and Tokenizer Setup ---
    print("\nLoading models and tokenizers...")
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    circuit_model = CircuitDiscoveryGPT2.from_pretrained_with_pruning(cfg.model_name, pruning_config).to(cfg.device).eval()
    full_model = GPT2LMHeadModel.from_pretrained(cfg.model_name).to(cfg.device).eval()
    for param in full_model.parameters(): 
        param.requires_grad = False

    disable_dropout(circuit_model)
    
    # Freeze base model weights and unfreeze gate parameters
    total_params = 0
    trainable_params = 0
    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        if 'gate' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params += param.numel()
            
    print(f"Total params: {total_params}, Trainable gate params: {trainable_params} ({(trainable_params/total_params*100):.4f}%)")

    # --- Dataset Setup ---
    print("\nSetting up IOI dataset...")
    test_data = load_or_generate_ioi_data(split="test", num_samples=1000) 
    train_data = load_or_generate_ioi_data(split="train", num_samples=200)
    val_data = load_or_generate_ioi_data(split="validation", num_samples=200)

    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, cfg.device, batch_size=cfg.batch_size)
    test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, cfg.device, batch_size=cfg.batch_size)

    train_dataset = IOIDataset(train_data, tokenizer)
    val_dataset = IOIDataset(val_data, tokenizer)
    test_dataset = IOIDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # --- Baseline Evaluation ---
    print("\nRunning baseline evaluation on full model...")
    baseline_results = run_evaluation(full_model, "Baseline Full Model", None, test_dataloader, cfg.device, tokenizer)
    base_accuracy = baseline_results.get("accuracy", 0.0)
    
    # Initial Evaluation
    circuit_model.eval()
    initial_results = run_evaluation(circuit_model, "Initial Circuit Model", full_model, val_dataloader, cfg.device, tokenizer)

    # --- Training Setup ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-4)
    
    print(f"\nTargeting accuracy within {(cfg.accuracy_budget*100):.2f}% of baseline ({base_accuracy:.4f})")
    
    total_steps = 0
    start_time = time.time()
    
    best_val_sparsity = -1
    best_checkpoint_path = os.path.join(cfg.output_dir, "best_circuit_gates.pt")

    for epoch in range(cfg.num_epochs):
        circuit_model.train()
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            for key, val in batch.items():
                if isinstance(val, torch.Tensor): batch[key] = val.to(cfg.device)
            
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            with torch.no_grad():
                target_outputs = full_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            
            batch_size_curr = circuit_outputs.logits.size(0)
            total_kl = 0
            
            for i in range(batch_size_curr):
                t_start = batch['T_Start'][i].item()-1
                t_end = batch['T_End'][i].item()-1
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, valid_length)
                
                if t_start < end_pos:
                    clogits = circuit_outputs.logits[i, t_start]
                    tlogits = target_outputs.logits[i, t_start]
                    kl = F.kl_div(F.log_softmax(clogits, dim=-1), F.log_softmax(tlogits, dim=-1), reduction='sum', log_target=True)
                    total_kl += kl
            
            pos_good = batch['T_Start'] - 1
            pos_bad = batch['D_Start'] - 1
            token_good = batch['target_tokens'][:, 0]
            token_bad = batch['distractor_tokens'][:, 0]
            batch_indices = torch.arange(batch_size_curr, device=cfg.device)
            
            logit_good = circuit_outputs.logits[batch_indices, pos_good, token_good]
            logit_bad = circuit_outputs.logits[batch_indices, pos_bad, token_bad]
            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()
            
            kl_loss = total_kl / batch_size_curr
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            
            loss = kl_loss * 1 + sparsity_loss * 25 + task_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1
            
        scheduler.step()
        
        # Averages
        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)
        lr_current = scheduler.get_last_lr()[0]
        
        if cfg.use_wandb:
            wandb.log({
                "train/loss": avg_loss,
                "train/kl_loss": avg_kl,
                "train/sparsity_loss": avg_sparsity,
                "train/lr": lr_current,
                "epoch": epoch
            })

        print(f"Epoch {epoch+1}/{cfg.num_epochs} - Loss: {avg_loss:.4f} - Sp: {avg_sparsity:.4f} - LR: {lr_current:.2e}")

        # Automated Validation and Checkpointing Check
        if (epoch + 1) == cfg.num_epochs or (epoch + 1) % cfg.eval_every == 0:
            circuit_model.eval()
            print(f"--- Validating at Epoch {epoch+1} ---")
            val_results = run_evaluation(circuit_model, f"Val Ep {epoch+1}", full_model, val_dataloader, cfg.device, tokenizer)
            
            val_acc = val_results.get("accuracy", 0.0)
            
            # Simple heuristic: If accuracy is in budget, check if sparsity increased since last checkpoint
            # For simplicity, if we meet the budget target reasonably and sparsity is good, we save.
            acceptable_accuracy = base_accuracy - cfg.accuracy_budget
            if val_acc >= acceptable_accuracy:
                if avg_sparsity > best_val_sparsity:
                    print(f"[*] New Best Sparsity that meets accuracy constraints! Saving checkpoint to {best_checkpoint_path}...")
                    best_val_sparsity = avg_sparsity
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': {k: v for k, v in circuit_model.state_dict().items() if 'gate' in k},
                        'val_accuracy': val_acc,
                        'val_sparsity': avg_sparsity
                    }, best_checkpoint_path)

            if cfg.use_wandb:
                # Log all validation results returned by run_evaluation
                val_log_dict = {f"eval/{k}": v for k, v in val_results.items() if isinstance(v, (int, float))}
                val_log_dict["eval/acceptable_accuracy"] = acceptable_accuracy
                wandb.log(val_log_dict)
                
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

    # --- Option: Load best checkpoint if one was found ---
    if os.path.exists(best_checkpoint_path):
        print("\nLoading best checkpoint for final evaluation...")
        chkpt = torch.load(best_checkpoint_path, map_location=cfg.device)
        # Load only gate params into the current model
        circuit_model.load_state_dict(chkpt['model_state_dict'], strict=False)

    # --- Final Analysis ---
    print("\n--- Analyzing and finalizing circuit ---")
    analyze_and_finalize_circuit(circuit_model)
    
    # --- Final Evaluation on Test Set ---
    circuit_model.eval()
    print("\n--- Final evaluation on test set ---")
    final_results = run_evaluation(circuit_model, "Final Pruned Circuit", full_model, test_dataloader, cfg.device, tokenizer)
    
    if cfg.use_wandb:
        final_log_dict = {f"test/{k}": v for k, v in final_results.items() if isinstance(v, (int, float))}
        wandb.log(final_log_dict)
        wandb.finish()

if __name__ == '__main__':
    main()
