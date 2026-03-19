"""
Quick check: Are gates actually receiving gradients to OPEN (decrease KL)?

This will show:
1. Current gate log_alpha values
2. Gradient direction (should be positive to open gates)
3. Whether gates are learning
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama_circuit import PrunableLlamaForCausalLM
from ioi_llama_hybrid_adaptive import HybridLlamaPruningConfig
from dataset.gp_llama import GPDatasetLlama, load_or_generate_gp_data
from torch.utils.data import DataLoader
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Load models
print("\nLoading models...")
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pruning_config = HybridLlamaPruningConfig()
circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
    'meta-llama/Llama-3.2-1B',
    pruning_config,
    torch_dtype=torch.bfloat16
).to(DEVICE)

full_model = LlamaForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B',
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

# Freeze everything except gates
GATE_PATTERNS = ('_gates.', '_gate.', 'embedding_gate.', 'layer_gates.')
for name, param in circuit_model.named_parameters():
    is_gate = any(p in name for p in GATE_PATTERNS)
    param.requires_grad = is_gate
    if is_gate:
        param.data = param.data.float()

# Get one batch
print("\nLoading data...")
train_data = load_or_generate_gp_data(split="train", num_samples=16)
dataset = GPDatasetLlama(train_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(dataloader))
batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# Cache full model output
print("\nCaching full model output...")
with torch.no_grad():
    full_output = full_model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        use_cache=False
    )

# Sample a few gates before training
print("\n" + "="*80)
print("INITIAL GATE VALUES")
print("="*80)
sample_gates = {}
circuit_model.eval()
with torch.no_grad():
    if hasattr(circuit_model, 'embedding_gate'):
        gate_val = circuit_model.embedding_gate().cpu().item()
        log_alpha_val = circuit_model.embedding_gate.log_alpha.cpu().item()
        sample_gates['embedding'] = (log_alpha_val, gate_val)
        print(f"Embedding gate: log_alpha={log_alpha_val:.3f}, gate={gate_val:.3f}")

    # First layer, first head
    first_layer = circuit_model.model.layers[0]
    if hasattr(first_layer.attn, 'head_gates') and first_layer.attn.head_gates:
        log_alpha_val = first_layer.attn.head_gates.log_alpha[0].cpu().item()
        gate_val = first_layer.attn.head_gates()[0].cpu().item()
        sample_gates['head_0'] = (log_alpha_val, gate_val)
        print(f"Layer 0, Head 0: log_alpha={log_alpha_val:.3f}, gate={gate_val:.3f}")

# Forward pass in TRAINING mode
print("\n" + "="*80)
print("COMPUTING KL LOSS (TRAINING MODE)")
print("="*80)
circuit_model.train()
outputs = circuit_model(
    input_ids=batch['input_ids'],
    corrupted_input_ids=batch['corrupted_input_ids'],
    attention_mask=batch['attention_mask'],
    use_cache=False,
)

# Compute KL loss
total_kl = 0
batch_size = outputs.logits.size(0)
for i in range(batch_size):
    pred_pos = batch['prefix_length'][i].item() - 1
    valid_length = batch['attention_mask'][i].sum().item()

    if pred_pos < valid_length:
        kl = F.kl_div(
            F.log_softmax(outputs.logits[i, pred_pos].float(), dim=-1),
            F.log_softmax(full_output.logits[i, pred_pos].float(), dim=-1),
            reduction='sum',
            log_target=True
        )
        total_kl += kl

kl_loss = total_kl / batch_size
print(f"KL Loss: {kl_loss.item():.6f}")

# Backward
print("\nComputing gradients...")
kl_loss.backward()

# Check gradients
print("\n" + "="*80)
print("GRADIENT ANALYSIS")
print("="*80)

if hasattr(circuit_model, 'embedding_gate'):
    grad = circuit_model.embedding_gate.log_alpha.grad
    if grad is not None:
        print(f"Embedding gate gradient: {grad.item():.6f}")
        print(f"  -> Gradient direction: {'OPEN gates (good!)' if grad.item() > 0 else 'CLOSE gates (bad!)'}")

first_layer = circuit_model.model.layers[0]
if hasattr(first_layer.attn, 'head_gates') and first_layer.attn.head_gates:
    grad = first_layer.attn.head_gates.log_alpha.grad[0]
    if grad is not None:
        print(f"Layer 0, Head 0 gradient: {grad.item():.6f}")
        print(f"  -> Gradient direction: {'OPEN gates (good!)' if grad.item() > 0 else 'CLOSE gates (bad!)'}")

# Collect all gate gradients
print("\n" + "="*80)
print("ALL GATE GRADIENT STATISTICS")
print("="*80)

all_grads = []
positive_grads = 0
negative_grads = 0
zero_grads = 0

for name, param in circuit_model.named_parameters():
    if param.requires_grad and param.grad is not None and 'gate' in name:
        grad_vals = param.grad.flatten()
        all_grads.extend(grad_vals.cpu().tolist())
        positive_grads += (grad_vals > 1e-8).sum().item()
        negative_grads += (grad_vals < -1e-8).sum().item()
        zero_grads += ((grad_vals >= -1e-8) & (grad_vals <= 1e-8)).sum().item()

if all_grads:
    all_grads_tensor = torch.tensor(all_grads)
    print(f"Total gate parameters with gradients: {len(all_grads)}")
    print(f"  Positive gradients (open gates): {positive_grads} ({positive_grads/len(all_grads)*100:.1f}%)")
    print(f"  Negative gradients (close gates): {negative_grads} ({negative_grads/len(all_grads)*100:.1f}%)")
    print(f"  Near-zero gradients: {zero_grads} ({zero_grads/len(all_grads)*100:.1f}%)")
    print(f"\nGradient magnitude:")
    print(f"  Mean: {all_grads_tensor.abs().mean():.6f}")
    print(f"  Median: {all_grads_tensor.abs().median():.6f}")
    print(f"  Max: {all_grads_tensor.abs().max():.6f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("""
Expected behavior (KL-only training):
- Most gradients should be POSITIVE (want to open gates to match full model)
- Gradient magnitude should be reasonable (not too small)
- KL loss should DECREASE over time

If you see:
- Negative gradients → Gates closing → KL increasing ❌
- Tiny gradients (< 1e-6) → No learning ❌
- Mixed gradients → Conflicting signals (possible issue) ⚠️
""")
