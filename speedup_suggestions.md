# Speedup Suggestions for Llama Gate Training

## High Impact (Easy)

### 1. Cache Full Model Outputs
The biggest bottleneck: **every batch runs the full 1.2B model forward just to get target logits**, even though it's frozen. Pre-compute once and reuse.

```python
# Before training loop — cache all full model outputs
print("Pre-computing full model outputs...")
cached_outputs = {}
with torch.no_grad():
    for batch_idx, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)
        out = full_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        cached_outputs[batch_idx] = out.logits.cpu()  # store on CPU to save GPU mem

# In training loop, replace full_model forward with:
target_logits = cached_outputs[batch_idx].to(DEVICE)
```
**Expected speedup: ~40-50%** (eliminates half the compute per step)

---

### 2. Use `torch.no_grad()` for the Corrupted Stream
The corrupted stream doesn't need gradients — only the gates do. Wrap corrupted embeddings with `detach()`:

```python
# In PrunableLlamaForCausalLM.forward(), after embedding:
hidden_states_corrupted = corrupted_inputs_embeds.detach().clone()
```

**Expected speedup: ~10-15%** (reduces backward graph size)

---

### 3. `torch.compile()` the Circuit Model
PyTorch 2.x's compiler can fuse operations and reduce kernel launch overhead:

```python
circuit_model = torch.compile(circuit_model, mode="reduce-overhead")
```

> [!WARNING]
> May require `dynamic=True` if batch sizes vary. Test with `--dry-run` first.

**Expected speedup: ~15-30%**

---

### 4. AMP (Automatic Mixed Precision) for the Training Loop
Use autocast for the forward pass to keep most computation in bfloat16:

```python
scaler = torch.amp.GradScaler('cuda')

for batch in train_dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        circuit_outputs = circuit_model(...)
        loss = kl_loss * 1.5 + sparsity_loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Expected speedup: ~10-20%**

---

## Medium Impact

### 5. Gradient Accumulation (trade speed for quality)
Use smaller micro-batches to reduce memory, then accumulate:

```python
ACCUM_STEPS = 4
MICRO_BATCH = BATCH_SIZE // ACCUM_STEPS

for step, batch in enumerate(train_dataloader):
    loss = compute_loss(batch) / ACCUM_STEPS
    loss.backward()
    if (step + 1) % ACCUM_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Frees GPU memory → allows **larger effective batch size** or **longer sequences**.

---

### 6. Reduce Validation Frequency
Currently validating every 10 epochs. For early training, every 25-50 is fine:

```python
if (epoch + 1) % 50 == 0 or epoch == NUM_EPOCHS - 1:
    # run validation
```

---

### 7. Gradient Checkpointing
Trade compute for memory — recompute activations during backward instead of storing them:

```python
circuit_model.gradient_checkpointing_enable()
```

Doesn't speed up per-step, but **allows larger batch sizes** which improves GPU utilization.

---

## Lower Impact (but free)

### 8. Pin Memory in DataLoaders
```python
train_dataloader = DataLoader(..., pin_memory=True, num_workers=2)
```

### 9. Set `use_cache=False`
KV cache is unnecessary for training (fixed-length sequences):
```python
circuit_outputs = circuit_model(..., use_cache=False)
```

### 10. Early Stopping
Monitor sparsity and stop when gates have converged:
```python
if avg_sparsity < 0.01 and epoch > 100:
    print("Gates converged, stopping early")
    break
```

---

## Summary — Priority Order

| # | Optimization | Effort | Speedup |
|---|---|---|---|
| 1 | Cache full model outputs | Low | **40-50%** |
| 2 | `detach()` corrupted stream | Trivial | 10-15% |
| 3 | `torch.compile()` | One line | 15-30% |
| 4 | AMP autocast | Low | 10-20% |
| 9 | `use_cache=False` | One line | ~5% |

Applying #1 + #2 + #9 alone could cut training time roughly in half.
