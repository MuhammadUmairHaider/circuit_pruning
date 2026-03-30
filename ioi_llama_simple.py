import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import random
import time
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from dataset.ioi_llama import IOIDatasetLlama, generate_ioi_data_llama, run_evaluation, filter_dataset_by_model_correctness
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION
# ==============================================================================
from dataclasses import dataclass

PRUNING_FACTOR = 1.0
@dataclass
# class LlamaPruningConfig(PruningConfig):
#     # Start with gates FULLY OPEN (log_alpha > 0) so gradient flows immediately
#     init_value: float = 0.5

#     # CRITICAL: Don't prune for the first ~5-10 epochs
#     sparsity_warmup_steps: int = 1000

#     depth_penalty_scaling: float = 0.0

#     # 1. Heads: Llama-1B has 32 heads per layer, 16 layers
#     prune_attention_heads: bool = True
#     lambda_attention_heads: float = 0.8

#     # 2. Neurons (Hidden): Llama-1B has 8192 intermediate neurons
#     prune_mlp_hidden: bool = True
#     lambda_mlp_hidden: float = 1.0

#     # 3. MLP Output (Residual)
#     prune_mlp_output: bool = True
#     lambda_mlp_output: float = 1.0

#     # 4. Attention Neurons
#     prune_attention_neurons: bool = True
#     lambda_attention_neurons: float = 0.15

#     # Structure pruning (Blocks/Layers)
#     prune_attention_blocks: bool = True
#     lambda_attention_blocks: float = 0.5

#     prune_mlp_blocks: bool = True
#     lambda_mlp_blocks: float = 0.5

#     prune_full_layers: bool = False
#     lambda_full_layers: float = 0.0

#     prune_embedding: bool = False
#     lambda_embedding: float = 1.0 * PRUNING_FACTOR


class LlamaPruningConfig(PruningConfig):
    # Start with gates FULLY OPEN (log_alpha > 0) so gradient flows immediately
    init_value: float = 0.5

    # CRITICAL: Don't prune for the first ~5-10 epochs
    sparsity_warmup_steps: int = 100

    depth_penalty_scaling: float = 0.0

    # 1. Heads: Llama-1B has 32 heads per layer, 16 layers
    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0

    # 2. Neurons (Hidden): Llama-1B has 8192 intermediate neurons
    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0

    # 3. MLP Output (Residual)
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0

    # 4. Attention Neurons
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0

    # Structure pruning (Blocks/Layers)
    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 1.0

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0

    prune_embedding: bool = False
    lambda_embedding: float = 1.0 * PRUNING_FACTOR


# ==============================================================================
# MAIN EXECUTION FOR IOI TASK
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_NAME = 'meta-llama/Llama-3.2-1B'
    NUM_EPOCHS = 500

    LEARNING_RATE = 3e-2
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 64
    ACCURACY_BUDGET = 0.05
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Read HF token ---
    import os
    hf_token = None
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_tokken.txt")
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            hf_token = f.read().strip()

    pruning_config = LlamaPruningConfig()

    # --- Model and Tokenizer Setup ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}

    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL_NAME, pruning_config, **model_kwargs
    ).to(DEVICE).eval()

    full_model = LlamaForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs).to(DEVICE).eval()
    for param in full_model.parameters(): param.requires_grad = False

    # --- Disable all built-in dropout layers in the circuit model ---
    print("\n--- Disabling all built-in dropout layers in the circuit model ---")
    disable_dropout(circuit_model)

    # --- Freeze the base model and unfreeze only the gates ---
    # NOTE: Can't use 'gate' in name — Llama has gate_proj (SwiGLU). Use specific gate patterns.
    print("Freezing base model weights and unfreezing gate parameters...")
    GATE_PATTERNS = ('_gates.', '_gate.', 'embedding_gate.', 'layer_gates.')
    total_params = 0
    trainable_params = 0
    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        if not any(p in name for p in GATE_PATTERNS):
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()  # Gates in float32 for stable training
            trainable_params += param.numel()

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\nSetting up IOI dataset...")

    all_data = generate_ioi_data_llama(num_samples=1400, tokenizer=tokenizer, seed=42)
    train_data = all_data[:200]
    val_data   = all_data[200:400]
    test_data  = all_data[400:1400]

    print("\n--- Filtering datasets based on Base Model correctness ---")
    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)
    test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE)

    train_dataset = IOIDatasetLlama(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset   = IOIDatasetLlama(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset  = IOIDatasetLlama(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    # --- Pre-cache full model outputs, then offload to CPU to free GPU memory ---
    print("\nPre-caching full model outputs for training data...")
    cached_train_logits = {}
    full_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Caching full model")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)
            out = full_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )
            cached_train_logits[batch_idx] = out.logits.detach()
    full_model = full_model.cpu()
    torch.cuda.empty_cache()
    print(f"Cached {len(cached_train_logits)} batches. Full model offloaded to CPU.")

    # --- Training ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)

    print(f"\n--- Starting training to find 'Indirect Object Identification' circuit ---")
    print(f"Target: Maintain accuracy within {ACCURACY_BUDGET*100}% of baseline ({base_accuracy:.4f})")

    circuit_model.train()
    total_steps = 0

    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_sparsity_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)

            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )

            # Use pre-cached full model outputs
            target_logits_cached = cached_train_logits[batch_idx]

            # Calculate KL loss
            batch_size_curr = circuit_outputs.logits.size(0)
            total_kl = 0

            for i in range(batch_size_curr):
                t_start = batch['T_Start'][i].item() - 1
                t_end = batch['T_End'][i].item() - 1

                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, int(valid_length))

                if t_start < end_pos:
                    circuit_logits = circuit_outputs.logits[i, t_start].float()
                    target_logits = target_logits_cached[i, t_start].float()

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

            logit_good = circuit_outputs.logits[batch_indices, pos_good, token_good].float()
            logit_bad = circuit_outputs.logits[batch_indices, pos_bad, token_bad].float()

            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()

            kl_loss = total_kl / batch_size_curr
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']

            # Total loss
            loss = kl_loss * 1.0 + sparsity_loss * 15.0  # + task_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        avg_loss = epoch_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_sparsity = epoch_sparsity_loss / len(train_dataloader)

        epoch_pbar.set_postfix({
            'L': f"{avg_loss:.3f}",
            'Sp': f"{avg_sparsity:.3f}",
            'Time': f"{epoch_duration:.2f}s"
        })

        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            full_model = full_model.to(DEVICE)
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Val Ep {epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_dataloader,
                device=DEVICE,
                tokenizer=tokenizer
            )
            full_model = full_model.cpu()
            torch.cuda.empty_cache()
            circuit_model.train()

    # --- Final Analysis and Pruning ---
    print("\n--- Analyzing and finalizing circuit ---")
    analyze_and_finalize_circuit(circuit_model)

    # --- Final Evaluation on Test Set ---
    print("\n--- Final evaluation on test set ---")
    circuit_model.eval()
    full_model = full_model.to(DEVICE)
    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Final Pruned Circuit",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer
    )

    # --- Save masks and test dataset for post-hoc analysis ---
    import pickle
    import os
    from models.l0 import HardConcreteGate

    SAVE_DIR = 'results/ioi_llama_simple'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Save gate log_alpha state dict
    gate_state = {}
    for name, module in circuit_model.named_modules():
        if isinstance(module, HardConcreteGate):
            gate_state[name] = module.log_alpha.detach().cpu().clone()
    torch.save(gate_state, os.path.join(SAVE_DIR, 'gate_log_alphas.pt'))
    print(f"  Saved gate log_alphas: {len(gate_state)} gates -> {SAVE_DIR}/gate_log_alphas.pt")

    # 2. Save binary masks
    binary_masks = {name: (la > 0).float() for name, la in gate_state.items()}
    torch.save(binary_masks, os.path.join(SAVE_DIR, 'binary_masks.pt'))
    sparsity = 1.0 - sum(m.sum().item() for m in binary_masks.values()) / sum(m.numel() for m in binary_masks.values())
    print(f"  Saved binary masks: overall sparsity={sparsity:.4f} -> {SAVE_DIR}/binary_masks.pt")

    # 3. Save raw test data for notebook reconstruction
    with open(os.path.join(SAVE_DIR, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    print(f"  Saved test data: {len(test_data)} samples -> {SAVE_DIR}/test_data.pkl")

    # 4. Save run config for notebook reference
    run_config = {
        'model': MODEL_NAME,
        'save_dir': SAVE_DIR,
        'num_epochs': NUM_EPOCHS,
        'device': DEVICE,
        'accuracy_budget': ACCURACY_BUDGET,
        'base_accuracy': base_accuracy,
        'final_accuracy': final_results.get('accuracy', 0.0),
        'final_sparsity': sparsity,
    }
    with open(os.path.join(SAVE_DIR, 'run_config.pkl'), 'wb') as f:
        pickle.dump(run_config, f)
    print(f"  Saved run config -> {SAVE_DIR}/run_config.pkl")
