"""
Evaluate the saved IOI circuit for Llama-3.2-1B at KL budget 0.5.

Loads the model from results/ioi_kl_budget_final_1B_with_blocks_0.5kl,
runs evaluation on the saved test data, and prints circuit analysis.
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from models.l0 import HardConcreteGate
from dataset.ioi_llama import IOIDatasetLlama, run_evaluation
from utils import disable_dropout, analyze_and_finalize_circuit
from dataclasses import dataclass


# ==============================================================================
# PRUNING CONFIG (matches the one used during training)
# ==============================================================================

@dataclass
class KLBudgetLlamaPruningConfig(PruningConfig):
    init_value: float = 2.0
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.0

    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0

    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 5.0

    prune_mlp_output: bool = True
    lambda_mlp_output: float = 5.0

    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0

    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.00001

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.00001

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    RESULTS_DIR = './results/ioi_kl_budget_final_1B_with_blocks_0.5kl'
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read run config
    with open(os.path.join(RESULTS_DIR, 'run_config.pkl'), 'rb') as f:
        run_config = pickle.load(f)
    MODEL_NAME = run_config['model']
    print(f"Model: {MODEL_NAME}")
    print(f"KL Budget: {run_config['kl_budget']}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Device: {DEVICE}")

    # Read HF token
    hf_token = None
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_tokken.txt')
    if os.path.exists(token_file):
        with open(token_file) as f:
            hf_token = f.read().strip()

    # --- Tokenizer ---
    print("\n--- Loading tokenizer and models ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}

    # --- Circuit model ---
    pruning_config = KLBudgetLlamaPruningConfig()
    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL_NAME, pruning_config, **model_kwargs
    ).to(DEVICE).eval()
    disable_dropout(circuit_model)

    # --- Load saved gate log_alphas ---
    gate_log_alphas_path = os.path.join(RESULTS_DIR, 'gate_log_alphas.pt')
    print(f"\n--- Loading gate log_alphas from {gate_log_alphas_path} ---")
    gate_log_alphas = torch.load(gate_log_alphas_path, map_location=DEVICE, weights_only=False)

    gate_modules = {
        name: m for name, m in circuit_model.named_modules()
        if isinstance(m, HardConcreteGate)
    }
    loaded = 0
    for name, log_alpha in gate_log_alphas.items():
        if name in gate_modules:
            gate_modules[name].log_alpha.data.copy_(log_alpha.float().to(DEVICE))
            loaded += 1
    print(f"Loaded {loaded}/{len(gate_log_alphas)} gate log_alphas.")

    # --- Full model (for faithfulness evaluation) ---
    print("\n--- Loading full model for faithfulness evaluation ---")
    full_model = LlamaForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    # --- Load saved test data ---
    test_data_path = os.path.join(RESULTS_DIR, 'test_data.pkl')
    print(f"\n--- Loading test data from {test_data_path} ---")
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    print(f"Test samples: {len(test_data)}")

    test_dataset = IOIDatasetLlama(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Baseline evaluation ---
    print("\n--- Baseline evaluation (full model) ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline Full Model",
        full_model_for_faithfulness=None,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    # --- Circuit model evaluation ---
    print("\n--- Circuit model evaluation ---")
    circuit_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Circuit (KL=0.5)",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    # --- Circuit analysis ---
    print("\n--- Circuit analysis ---")
    analyze_and_finalize_circuit(circuit_model)
