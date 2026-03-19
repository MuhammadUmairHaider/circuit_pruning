"""
Diagnostic script to understand why KL loss isn't decreasing.

Run this to check:
1. Gate initialization values
2. Gradient magnitudes
3. Cache consistency
4. KL computation correctness
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama_circuit import PrunableLlamaForCausalLM
from ioi_llama_hybrid_adaptive import HybridLlamaPruningConfig
from dataset.gp_llama import GPDatasetLlama, load_or_generate_gp_data
from torch.utils.data import DataLoader
import os

# Suppress warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load model
print("\n1. Loading model...")
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pruning_config = HybridLlamaPruningConfig()
circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
    'meta-llama/Llama-3.2-1B',
    pruning_config,
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

full_model = LlamaForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B',
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

# Freeze base, unfreeze gates
GATE_PATTERNS = ('_gates.', '_gate.', 'embedding_gate.', 'layer_gates.')
for name, param in circuit_model.named_parameters():
    is_gate = any(p in name for p in GATE_PATTERNS)
    param.requires_grad = is_gate
    if is_gate:
        param.data = param.data.float()

# Check gate initialization
print("\n2. Checking gate initialization...")
circuit_model.eval()
gate_values = []
for name, module in circuit_model.named_modules():
    if hasattr(module, 'log_alpha'):
        with torch.no_grad():
            log_alpha = module.log_alpha.cpu()
            gate_output = module().cpu()
            print(f"  {name:40s} log_alpha: {log_alpha.mean():.3f}±{log_alpha.std():.3f}, "
                  f"gate: {gate_output.mean():.3f}±{gate_output.std():.3f}")
            gate_values.extend(gate_output.flatten().tolist())

print(f"\n  Overall gate statistics:")
print(f"    Mean: {torch.tensor(gate_values).mean():.4f}")
print(f"    Std:  {torch.tensor(gate_values).std():.4f}")
print(f"    Min:  {torch.tensor(gate_values).min():.4f}")
print(f"    Max:  {torch.tensor(gate_values).max():.4f}")

# Load small dataset
print("\n3. Loading dataset...")
train_data = load_or_generate_gp_data(split="train", num_samples=16)
train_dataset = GPDatasetLlama(train_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)  # NO SHUFFLE for testing

# Get one batch
batch = next(iter(train_dataloader))
batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print(f"  Batch size: {batch['input_ids'].shape[0]}")
print(f"  Sequence length: {batch['input_ids'].shape[1]}")
print(f"  Prefix lengths: {batch['prefix_length'].tolist()}")

# Test forward pass
print("\n4. Testing forward pass...")
circuit_model.train()  # Set to training mode
with torch.enable_grad():
    outputs = circuit_model(
        input_ids=batch['input_ids'],
        corrupted_input_ids=batch['corrupted_input_ids'],
        attention_mask=batch['attention_mask'],
        use_cache=False,
    )
    print(f"  Output logits shape: {outputs.logits.shape}")
    print(f"  Output logits dtype: {outputs.logits.dtype}")

# Get full model outputs (for KL target)
print("\n5. Getting full model outputs...")
with torch.no_grad():
    full_outputs = full_model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        use_cache=False
    )
    print(f"  Full model logits shape: {full_outputs.logits.shape}")

# Compute KL loss (replicate training code)
print("\n6. Computing KL loss...")
total_kl = 0
batch_size = outputs.logits.size(0)
for i in range(batch_size):
    pred_pos = batch['prefix_length'][i].item() - 1
    valid_length = batch['attention_mask'][i].sum().item()

    print(f"\n  Sample {i}:")
    print(f"    Prefix length: {batch['prefix_length'][i].item()}")
    print(f"    Pred position: {pred_pos}")
    print(f"    Valid length: {valid_length}")

    if pred_pos < valid_length:
        circuit_logits = outputs.logits[i, pred_pos].float()
        full_logits = full_outputs.logits[i, pred_pos].float()

        # Check top-5 tokens
        circuit_top5 = torch.topk(circuit_logits, 5)
        full_top5 = torch.topk(full_logits, 5)

        print(f"    Circuit top-5 tokens: {[tokenizer.decode(t) for t in circuit_top5.indices]}")
        print(f"    Circuit top-5 logits: {circuit_top5.values.tolist()}")
        print(f"    Full top-5 tokens: {[tokenizer.decode(t) for t in full_top5.indices]}")
        print(f"    Full top-5 logits: {full_top5.values.tolist()}")

        # Compute KL
        kl = F.kl_div(
            F.log_softmax(circuit_logits, dim=-1),
            F.log_softmax(full_logits, dim=-1),
            reduction='sum',
            log_target=True
        )
        total_kl += kl
        print(f"    KL divergence: {kl.item():.4f}")

kl_loss = total_kl / batch_size
print(f"\n  Average KL loss: {kl_loss.item():.4f}")

# Test gradient flow
print("\n7. Testing gradient flow...")
kl_loss.backward()

gate_grad_stats = []
for name, param in circuit_model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_max = param.grad.abs().max().item()
        if 'gate' in name or 'log_alpha' in name:
            print(f"  {name:50s} grad_norm: {grad_norm:.6f}, mean: {grad_mean:.6f}, max: {grad_max:.6f}")
            gate_grad_stats.append(grad_norm)

if gate_grad_stats:
    print(f"\n  Gate gradient statistics:")
    print(f"    Mean norm: {torch.tensor(gate_grad_stats).mean():.6f}")
    print(f"    Max norm:  {torch.tensor(gate_grad_stats).max():.6f}")
    print(f"    Min norm:  {torch.tensor(gate_grad_stats).min():.6f}")
else:
    print("\n  ⚠️  WARNING: No gate gradients found!")

# Test cache consistency
print("\n8. Testing cache consistency with shuffle...")
dataloader_shuffle = DataLoader(train_dataset, batch_size=4, shuffle=True)
dataloader_no_shuffle = DataLoader(train_dataset, batch_size=4, shuffle=False)

print("  First batch (shuffle=True):")
batch1 = next(iter(dataloader_shuffle))
print(f"    First input_id: {batch1['input_ids'][0, :5].tolist()}")

print("  First batch (shuffle=False):")
batch2 = next(iter(dataloader_no_shuffle))
print(f"    First input_id: {batch2['input_ids'][0, :5].tolist()}")

print("  ⚠️  If these are different, shuffle is breaking your cache!")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
