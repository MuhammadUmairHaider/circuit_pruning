# IOI Circuit Discovery Training Guide

## Quick Start with Adaptive Training

### Basic Usage
```bash
# Standard adaptive training (recommended)
python ioi_llama_adaptive.py

# Quick test run
python ioi_llama_adaptive.py --dry-run

# More aggressive pruning
python ioi_llama_adaptive.py --target-sparsity 0.9

# With Flash Attention 2 (requires: pip install flash-attn)
python ioi_llama_adaptive.py --flash-attn
```

---

## What's Different in the Adaptive Version?

### 1. **Automatic Hyperparameter Tuning**
The adaptive scheduler automatically adjusts pruning pressure based on:
- **Current accuracy vs baseline**: If accuracy drops, reduce pruning
- **Current sparsity vs target**: If sparsity too low, increase pruning
- **Training stability**: Smooth adjustments via exponential moving average

**No more manual lambda tuning!** Just set your targets:
```bash
--target-sparsity 0.8      # Want 80% of gates pruned
--target-accuracy 0.95     # Want 95% of baseline accuracy
```

### 2. **Two Scheduler Options**

#### **Adaptive Scheduler** (Default - Recommended)
Dynamically adjusts based on validation metrics:
```python
# Monitors:
# - Validation accuracy every 10 epochs
# - Overall sparsity rate
# - KL divergence trend

# Adjusts:
# - Lambda multipliers for all gate types
# - Gradually increases/decreases pruning pressure
```

#### **Progressive Scheduler** (Simpler Alternative)
Fixed schedule with 3 phases:
```bash
python ioi_llama_adaptive.py --use-progressive
```
- Phase 1 (0-30%): Learn task with minimal pruning
- Phase 2 (30-70%): Gradual sparsity increase
- Phase 3 (70-100%): Aggressive pruning

### 3. **Early Stopping**
Training stops automatically when converged:
- Sparsity plateaued near target
- Accuracy stable and acceptable
- Saves time on long runs

### 4. **Training Visualization**
Automatically generates `training_dynamics.png`:
- Accuracy over time
- Sparsity progression
- KL divergence trend
- Lambda multiplier evolution

---

## Performance Optimizations

### Already Implemented ✅
1. ✅ **Pre-cached full model outputs** (40-50% speedup)
2. ✅ **Detached corrupted stream** (10-15% speedup)
3. ✅ **use_cache=False** (~5% speedup)
4. ✅ **Pin memory in DataLoaders**

### Easy Additions 🚀

#### Flash Attention 2 (30-50% speedup)
```bash
# Install (requires CUDA)
pip install flash-attn --no-build-isolation

# Use it
python ioi_llama_adaptive.py --flash-attn
```

#### Gradient Checkpointing (for larger models)
```python
# In ioi_llama_adaptive.py, after loading model:
circuit_model.gradient_checkpointing_enable()
```
Trades compute for memory → allows larger batch sizes.

#### torch.compile() (15-30% speedup)
```python
# After loading model:
circuit_model = torch.compile(circuit_model, mode="reduce-overhead")
```
⚠️ May have compatibility issues with custom forward pass. Test first!

---

## Understanding Training Dynamics

### Healthy Training Looks Like:

```
Epoch 50:  Acc: 0.92 | Sparsity: 0.35 | KL: 2.1
Epoch 100: Acc: 0.91 | Sparsity: 0.58 | KL: 1.8
Epoch 150: Acc: 0.90 | Sparsity: 0.72 | KL: 1.5
Epoch 200: Acc: 0.90 | Sparsity: 0.79 | KL: 1.4
```
- **Accuracy** gradually decreases but stays near target
- **Sparsity** steadily increases toward target
- **KL divergence** decreases (circuit becomes more faithful)

### Problem Signs:

#### 1. Accuracy Collapse
```
Epoch 50:  Acc: 0.85 | Sparsity: 0.15
Epoch 60:  Acc: 0.65 | Sparsity: 0.25  ← Too aggressive!
```
**Solution**:
- Lower `--target-sparsity` (e.g., 0.7 instead of 0.8)
- Increase `--target-accuracy` (e.g., 0.97 instead of 0.95)

#### 2. Sparsity Stalls
```
Epoch 100: Acc: 0.95 | Sparsity: 0.30
Epoch 200: Acc: 0.95 | Sparsity: 0.32  ← Not pruning enough!
```
**Solution**:
- Increase learning rate: `--lr 5e-2`
- Lower target accuracy: `--target-accuracy 0.93`
- Use progressive scheduler: `--use-progressive`

#### 3. Oscillation
```
Epoch 100: Acc: 0.92 | Sparsity: 0.60
Epoch 110: Acc: 0.85 | Sparsity: 0.70
Epoch 120: Acc: 0.91 | Sparsity: 0.55  ← Unstable!
```
**Solution**:
- Lower learning rate: `--lr 1e-2`
- Increase validation frequency (edit line in code: `if (epoch + 1) % 5 == 0`)

---

## Hyperparameter Tuning Guide

### Start Here (Defaults)
```bash
python ioi_llama_adaptive.py \
    --lr 3e-2 \
    --target-sparsity 0.8 \
    --target-accuracy 0.95 \
    --epochs 500
```

### More Aggressive Pruning
```bash
python ioi_llama_adaptive.py \
    --target-sparsity 0.9 \
    --target-accuracy 0.90 \
    --lr 5e-2
```

### Conservative (High Accuracy)
```bash
python ioi_llama_adaptive.py \
    --target-sparsity 0.7 \
    --target-accuracy 0.98 \
    --lr 2e-2
```

### Fast Iteration (Development)
```bash
python ioi_llama_adaptive.py \
    --dry-run \
    --epochs 50 \
    --flash-attn
```

---

## Advanced: Manual Lambda Tuning (Original Method)

If you want fine-grained control, edit `AdaptiveLlamaPruningConfig` in `ioi_llama_adaptive.py`:

```python
@dataclass
class AdaptiveLlamaPruningConfig(PruningConfig):
    # Fine-grained control over each component
    lambda_attention_heads: float = 0.8      # ← Heads very important
    lambda_attention_neurons: float = 0.15   # ← Neurons less important
    lambda_mlp_hidden: float = 1.0           # ← MLP hidden important
    lambda_mlp_output: float = 1.0           # ← MLP output important
    lambda_attention_blocks: float = 0.5     # ← Block-level pruning
    lambda_mlp_blocks: float = 0.5
```

**Rule of thumb**:
- **Higher lambda** → More aggressive pruning of that component
- **Lower lambda** → Preserve that component more

**Component importance for IOI** (from literature):
1. **Attention heads**: Very important (name movers, duplicate token heads)
2. **MLP neurons**: Important (backup/composition)
3. **Attention neurons**: Less critical (can prune more)

---

## Monitoring Training

### Real-time Monitoring
```bash
# In another terminal, watch checkpoints
watch -n 10 'ls -lh checkpoints_llama_ioi_adaptive/'

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Post-Training Analysis
```python
# Load training history
import torch
ckpt = torch.load('checkpoints_llama_ioi_adaptive/best_checkpoint.pt')
print(f"Best epoch: {ckpt['epoch']}")
print(f"Best accuracy: {ckpt['best_val_accuracy']:.4f}")

# View training dynamics plot
# Open: checkpoints_llama_ioi_adaptive/training_dynamics.png
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python ioi_llama_adaptive.py --batch-size 8

# Enable gradient checkpointing (add to code):
circuit_model.gradient_checkpointing_enable()
```

### Slow Training
```bash
# Enable all optimizations
python ioi_llama_adaptive.py --flash-attn --batch-size 32

# Reduce validation frequency (edit code):
if (epoch + 1) % 25 == 0:  # was 10
```

### Training Diverges
```bash
# Lower learning rate
python ioi_llama_adaptive.py --lr 1e-2

# Use progressive scheduler
python ioi_llama_adaptive.py --use-progressive
```

### Not Converging
```bash
# Run longer
python ioi_llama_adaptive.py --epochs 1000

# Or check if already converged (early stopping triggered)
# Look for: "Early stopping triggered - training converged!"
```

---

## Expected Training Times

**On A100 (80GB)**:
- Without Flash Attn: ~8-10 hours (500 epochs)
- With Flash Attn: ~5-6 hours (500 epochs)
- Early stop (typical): ~3-4 hours (~200-250 epochs)

**On A6000 (48GB)**:
- Without Flash Attn: ~12-15 hours (500 epochs)
- With Flash Attn: ~7-9 hours (500 epochs)

**On RTX 3090 (24GB)**:
- Batch size 8: ~15-20 hours (500 epochs)
- With Flash Attn: ~10-12 hours (500 epochs)

---

## Next Steps After Training

### 1. Evaluate Best Checkpoint
The script automatically evaluates the best checkpoint at the end.

### 2. Analyze Discovered Circuit
Check the printed output for:
- Which layers are pruned
- Which attention heads are active
- Which MLP neurons survived

### 3. Compare with Known IOI Circuit
The IOI circuit from literature includes:
- **Duplicate Token Heads** (detect repeated names)
- **Induction Heads** (pattern completion)
- **Name Mover Heads** (move correct name to output)
- **Backup Name Mover Heads** (redundancy)

Does your discovered circuit align?

### 4. Visualize Circuit
```python
# TODO: Add visualization script
python visualize_circuit.py --checkpoint checkpoints_llama_ioi_adaptive/best_checkpoint.pt
```

---

## FAQ

**Q: Should I use adaptive or progressive scheduler?**
A: Start with **adaptive** (default). It's more robust. Use progressive only if adaptive oscillates.

**Q: What's a good target sparsity?**
A: Start with **0.8** (80% pruned). IOI is a simple task, so high sparsity is achievable.

**Q: How do I know if my lambda values are good?**
A: With adaptive training, you don't need to tune lambdas! Just set target sparsity/accuracy.

**Q: Can I resume training?**
A: Yes! The script auto-resumes from the latest checkpoint by default.

**Q: Training is too slow, what's the single best optimization?**
A: **Flash Attention 2** (`--flash-attn`). 30-50% speedup with one flag.

**Q: My accuracy is lower than baseline, is that okay?**
A: Yes! A small accuracy drop (5-10%) is expected when pruning 80%+ of the model.

**Q: Can I use this for other tasks?**
A: Yes! Just replace the IOI dataset with your task's dataset. The adaptive scheduler is task-agnostic.
