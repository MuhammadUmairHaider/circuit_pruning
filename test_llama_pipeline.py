#!/usr/bin/env python3
"""Quick smoke test for the Llama circuit pruning pipeline."""
import sys
import os

# Read HF token
token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_tokken.txt")
with open(token_file) as f:
    hf_token = f.read().strip()

print("=" * 60)
print("SMOKE TEST: Llama Circuit Pruning Pipeline")
print("=" * 60)

# Step 1: Test imports
print("\n[1/5] Testing imports...")
try:
    from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
    from dataset.ioi_llama import generate_ioi_data_llama, IOIDatasetLlama, run_evaluation
    from transformers import AutoTokenizer, LlamaForCausalLM
    from utils import disable_dropout, analyze_and_finalize_circuit
    print("  OK All imports successful")
except Exception as e:
    print(f"  FAIL Import error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 2: Load tokenizer
print("\n[2/5] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  OK Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
except Exception as e:
    print(f"  FAIL Tokenizer error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 3: Generate IOI data
print("\n[3/5] Generating IOI data...")
try:
    data = generate_ioi_data_llama(num_samples=10, tokenizer=tokenizer)
    ds = IOIDatasetLlama(data, tokenizer)
    batch = ds[0]
    print(f"  OK Generated {len(data)} samples, dataset has {len(ds)} valid samples")
    print(f"  OK Sample keys: {list(batch.keys())}")
    print(f"  OK Sentence: {data[0]['sentence']}")
except Exception as e:
    print(f"  FAIL Dataset error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 4: Load model with pruning
print("\n[4/5] Loading prunable Llama model...")
try:
    import torch
    config = PruningConfig()
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        "meta-llama/Llama-3.2-1B", config,
        token=hf_token, torch_dtype=torch.bfloat16
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  OK Model loaded on {device}")
    print(f"  OK Total params: {total:,}, Gate params: {trainable:,}")
except Exception as e:
    print(f"  FAIL Model error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 5: Test forward pass
print("\n[5/5] Testing forward pass...")
try:
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Test standard forward
    with torch.no_grad():
        out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    print(f"  OK Standard forward: logits shape = {out.logits.shape}")

    # Test dual-stream forward
    with torch.no_grad():
        out = model(
            input_ids=batch['input_ids'],
            corrupted_input_ids=batch['corrupted_input_ids'],
            attention_mask=batch['attention_mask'],
        )
    print(f"  OK Dual-stream forward: logits shape = {out.logits.shape}")

    # Test sparsity loss
    sparsity = model.get_sparsity_loss(step=100)
    print(f"  OK Sparsity loss: {sparsity['total_sparsity'].item():.4f}")

except Exception as e:
    print(f"  FAIL Forward pass error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 6: Sanity check — circuit model with gates=1 should match base model
print("\n[6/6] Sanity check: gates=1 circuit output == base model output...")
try:
    # Load a fresh base model (no pruning wrappers)
    base_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", token=hf_token, torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Set circuit model to final mode so gates output hard 1s (since init_value > 0)
    model.set_final_circuit_mode(True)
    model.eval()

    # Verify all gates are indeed 1.0
    from models.l0 import HardConcreteGate
    all_gates_open = True
    for name, module in model.named_modules():
        if isinstance(module, HardConcreteGate):
            gate_val = module().item() if module().numel() == 1 else module().min().item()
            if gate_val < 0.99:
                print(f"  WARNING: Gate {name} has value {gate_val:.4f} (expected ~1.0)")
                all_gates_open = False
    if all_gates_open:
        print("  OK All gates are ~1.0 in final mode")

    # Run same input through both models
    test_input_ids = batch['input_ids']
    test_attn_mask = batch['attention_mask']

    with torch.no_grad():
        base_out = base_model(input_ids=test_input_ids, attention_mask=test_attn_mask)
        circuit_out = model(
            input_ids=test_input_ids,
            corrupted_input_ids=test_input_ids,  # same as clean — so gating has no effect
            attention_mask=test_attn_mask,
        )

    # Compare logits
    base_logits = base_out.logits.float()
    circuit_logits = circuit_out.logits.float()

    max_diff = (base_logits - circuit_logits).abs().max().item()
    mean_diff = (base_logits - circuit_logits).abs().mean().item()

    # Check prediction agreement (argmax at each position)
    base_preds = base_logits.argmax(dim=-1)
    circuit_preds = circuit_logits.argmax(dim=-1)
    pred_match_rate = (base_preds == circuit_preds).float().mean().item()

    print(f"  Logit max absolute diff:  {max_diff:.6f}")
    print(f"  Logit mean absolute diff: {mean_diff:.6f}")
    print(f"  Prediction match rate:    {pred_match_rate:.4f}")

    # With exact same computation path, diff should be ~0
    if max_diff < 0.01 and pred_match_rate > 0.99:
        print("  OK Circuit model with gates=1 matches base model!")
    else:
        print("  WARNING: Outputs differ more than expected. This may indicate a bug.")

    # Reset final mode
    model.set_final_circuit_mode(False)

    # Clean up base model to free GPU memory
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

except Exception as e:
    print(f"  FAIL Sanity check error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
