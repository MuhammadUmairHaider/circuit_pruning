# Quick Reference Cheat Sheet

## TL;DR - What Changed?

### Old Way (Manual Tuning)
```bash
# 1. Guess lambda values in code
lambda_attention_heads = 0.8  # Maybe?
lambda_mlp_hidden = 1.0       # Who knows?

# 2. Run training (6+ hours)
python ioi_llama.py

# 3. Check results
# - Accuracy good but sparsity low? Increase lambdas, try again
# - Sparsity good but accuracy bad? Decrease lambdas, try again
# - Repeat 3-5 times (18-30+ hours total!)
```

### New Way (Automatic)
```bash
# 1. Set targets (what you want)
python ioi_llama_adaptive.py \
    --target-sparsity 0.8 \    # Want 80% pruned
    --target-accuracy 0.95 \   # Want 95% of baseline
    --flash-attn               # 40% faster!

# 2. Training auto-adjusts and stops when done (~1-2 hours)
# 3. Done! ✨
```

---

## Commands Cheat Sheet

```bash
# Quick test (5 minutes)
python ioi_llama_adaptive.py --dry-run --flash-attn

# Standard run (recommended)
python ioi_llama_adaptive.py --flash-attn

# More aggressive pruning
python ioi_llama_adaptive.py --target-sparsity 0.9 --flash-attn

# Conservative (higher accuracy)
python ioi_llama_adaptive.py --target-sparsity 0.7 --target-accuracy 0.98 --flash-attn

# Use simpler scheduler
python ioi_llama_adaptive.py --use-progressive --flash-attn

# Continue from checkpoint (automatic)
python ioi_llama_adaptive.py --flash-attn
# (will auto-resume if checkpoints exist)

# Start fresh (ignore checkpoints)
python ioi_llama_adaptive.py --no-resume --flash-attn
```

---

## Speedups at a Glance

| Optimization | How to Enable | Speedup | Difficulty |
|-------------|---------------|---------|-----------|
| **Flash Attention** | `--flash-attn` | 30-50% | Easy (one flag) |
| Pre-cached outputs | ✅ Already done | 40-50% | Done |
| Detached corrupted | ✅ Already done | 10-15% | Done |
| torch.compile | Add to code | 15-30% | Medium (may break) |
| Early stopping | ✅ Built-in | 2-3x | Free |
| Larger batch size | `--batch-size 32` | 10-20% | Easy (needs memory) |

**Best combo**: `--flash-attn --batch-size 32` → ~2-3x faster

---

## Hyperparameter Quick Guide

### Target Sparsity
```
0.6 = 60% pruned (conservative)
0.7 = 70% pruned (safe default)
0.8 = 80% pruned (recommended)
0.9 = 90% pruned (aggressive)
```

### Target Accuracy
```
0.98 = 98% of baseline (very conservative)
0.95 = 95% of baseline (recommended)
0.90 = 90% of baseline (aggressive)
```

### Learning Rate
```
1e-2 = slow, stable
3e-2 = default (recommended)
5e-2 = fast, might oscillate
```

---

## Training Health Check

### ✅ Good Training
```
Epoch 100: Acc: 0.91 | Sparsity: 0.61
Epoch 150: Acc: 0.90 | Sparsity: 0.75
Epoch 200: Acc: 0.90 | Sparsity: 0.81
```
- Accuracy slowly decreasing
- Sparsity steadily increasing
- Both approaching targets

### ⚠️ Problem: Accuracy Collapse
```
Epoch 100: Acc: 0.85
Epoch 120: Acc: 0.65  ← Too low!
```
**Fix**: `--target-sparsity 0.7` (lower target)

### ⚠️ Problem: Sparsity Stalled
```
Epoch 200: Sparsity: 0.35
Epoch 300: Sparsity: 0.37  ← Not moving!
```
**Fix**: `--lr 5e-2` (higher learning rate)

### ⚠️ Problem: Oscillation
```
Epoch 100: Acc: 0.92 | Sparsity: 0.60
Epoch 110: Acc: 0.85 | Sparsity: 0.70
Epoch 120: Acc: 0.91 | Sparsity: 0.55  ← Bouncing!
```
**Fix**: `--lr 1e-2` (lower learning rate)

---

## File Structure

```
circuit_pruning/
├── ioi_llama.py                    # Original training script
├── ioi_llama_adaptive.py          # ⭐ NEW: Use this instead!
├── pruning_scheduler.py            # ⭐ NEW: Adaptive scheduler
│
├── models/
│   ├── llama_circuit.py           # LLaMA with pruning gates
│   └── l0.py                      # HardConcreteGate implementation
│
├── dataset/
│   └── ioi_llama.py              # IOI dataset for LLaMA
│
├── utils.py                       # Circuit analysis utilities
│
├── TRAINING_GUIDE.md              # ⭐ Comprehensive guide
├── IMPROVEMENTS_SUMMARY.md         # ⭐ What changed & why
├── QUICK_REFERENCE.md             # ⭐ This file
└── speedup_suggestions.md         # Detailed speedup analysis
```

---

## Troubleshooting One-Liners

```bash
# Out of memory
python ioi_llama_adaptive.py --batch-size 8 --flash-attn

# Too slow
python ioi_llama_adaptive.py --flash-attn --batch-size 32

# Training diverged
python ioi_llama_adaptive.py --lr 1e-2 --flash-attn

# Not converging after 500 epochs
python ioi_llama_adaptive.py --epochs 1000 --flash-attn

# Want to see what's happening
tail -f checkpoints_llama_ioi_adaptive/*.log  # (if you add logging)

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## Expected Results (Typical Run)

```bash
$ python ioi_llama_adaptive.py --flash-attn

Baseline Accuracy: 0.9520

[Training...]

Epoch 50:  Acc: 0.93 | Sparsity: 0.38
Epoch 100: Acc: 0.91 | Sparsity: 0.61
Epoch 150: Acc: 0.90 | Sparsity: 0.75
Epoch 200: Acc: 0.90 | Sparsity: 0.81

Early stopping triggered!

Final Results:
  Accuracy: 0.900 (baseline: 0.952)
  Sparsity: 0.812
  Prunable compression: 5.2x
  Time: 1.2 hours
```

---

## Next Steps After Training

1. **Check the circuit**:
   - See which layers survived
   - See which heads are active
   - Compare with known IOI circuit

2. **View training plot**:
   - `checkpoints_llama_ioi_adaptive/training_dynamics.png`

3. **Experiment**:
   - Try different sparsity targets
   - Compare adaptive vs progressive
   - Test on your own tasks

---

## Getting Help

1. **Read the guide**: `TRAINING_GUIDE.md`
2. **Check examples**: `IMPROVEMENTS_SUMMARY.md`
3. **Understand the code**: Inline comments in `pruning_scheduler.py`
4. **Compare versions**: `diff ioi_llama.py ioi_llama_adaptive.py`

---

## Installation (First Time Only)

```bash
# Required
pip install torch transformers tqdm

# Recommended (for Flash Attention)
pip install flash-attn --no-build-isolation

# Optional (for visualization)
pip install matplotlib
```

---

## One-Sentence Summary

**Instead of manually tuning lambda values 3-5 times (18-30 hours), just set target sparsity/accuracy and let the adaptive scheduler find the right values automatically (1-2 hours with Flash Attention).**
