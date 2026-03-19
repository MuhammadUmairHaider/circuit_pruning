# 🚀 Ready-to-Run Commands for Allocation Box

## Current Status: ✅ FIXED
Your environment now has:
- ✅ `torch 2.10.0`
- ✅ `torchvision 0.25.0`
- ✅ `torchaudio 2.10.0`
- ✅ `transformers 5.2.0`
- ⚠️ `flash-attn` - needs to be installed in allocation (with GPU)

---

## Step-by-Step Guide

### 1️⃣ Request Allocation (12 hours)
```bash
salloc --account=def-hsajjad --nodes=1 --gres=gpu:l40:1 --cpus-per-task=12 --mem=128000 --time=12:00:00
```

### 2️⃣ Load Modules
```bash
module load StdEnv/2023 python-build-bundle/2025a gcc arrow
```

### 3️⃣ Activate Environment
```bash
source /home/dogar/project_env/bin/activate
cd /gpfs/project/6075961/dogar/circuit_latest/circuit_pruning
```

### 4️⃣ Install Flash Attention (Optional but Recommended)

Flash attention needs GPU to compile, so install it in the allocation:

```bash
# This will take ~10-20 minutes to compile
pip install flash-attn --no-build-isolation

# Verify it works
python -c "import flash_attn; print(f'✓ Flash Attention: {flash_attn.__version__}')"
```

**If flash-attn installation fails or takes too long**, skip it and run without the `--flash-attn` flag (see option B below).

### 5️⃣ Run Your Experiment

#### **Option A: With Flash Attention (2-3x faster) ⚡**
```bash
python ioi_llama_hybrid_adaptive.py \
    --target-accuracy 0.95 \
    --flash-attn \
    --epochs 500 \
    --batch-size 16 \
    --lr 3e-2 \
    --save-dir checkpoints_llama_hybrid
```

#### **Option B: Without Flash Attention (safer, slower) 🐢**
```bash
python ioi_llama_hybrid_adaptive.py \
    --target-accuracy 0.95 \
    --epochs 500 \
    --batch-size 16 \
    --lr 3e-2 \
    --save-dir checkpoints_llama_hybrid
```

### 6️⃣ Quick Test First (Dry Run - 2 minutes)

**Always test before the full run:**
```bash
# Test with flash-attn
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn --dry-run

# Or test without flash-attn
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --dry-run
```

If the dry run completes successfully, you're good to go for the full run!

---

## 💡 Pro Tips

### Monitor Training Progress
The script will print updates every 10 epochs. Look for:
- **Accuracy**: Should stay near target (0.95 * baseline)
- **Sparsity**: Should increase over time
- **Phase**: Shows what the scheduler is doing

### Save Your Work
- Checkpoints saved to: `checkpoints_llama_hybrid/`
- Training plot: `checkpoints_llama_hybrid/training.png`

### If Something Goes Wrong

**Import Error:**
```bash
# Verify packages
pip list | grep -E "(torch|flash|transformers)"

# Should see:
# torch         2.10.0
# torchvision   0.25.0
# transformers  5.2.0
```

**Flash Attention Issues:**
Just remove the `--flash-attn` flag and run without it.

**Out of Memory:**
Reduce batch size:
```bash
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --batch-size 8
```

**Taking Too Long:**
Reduce epochs for initial testing:
```bash
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --epochs 100
```

---

## 📊 Expected Runtime

- **With flash-attn**: ~6-8 hours for 500 epochs
- **Without flash-attn**: ~12-16 hours for 500 epochs
- **Dry run**: ~2-3 minutes

---

## 🎯 What to Expect

The hybrid adaptive scheduler will:
1. Start with low sparsity (warmup phase)
2. Gradually increase sparsity while monitoring accuracy
3. Adjust lambda multiplier to maintain target accuracy (95% of baseline)
4. Converge when optimal sparsity is found
5. Generate a training plot showing the full process

**Target Outcome:**
- Accuracy: ~95% of baseline (as specified)
- Sparsity: 70-85% (auto-discovered)
- Clear training dynamics plot

---

## ✅ Final Checklist

Before starting your 12-hour run:

- [ ] In allocation with GPU (`nvidia-smi` works)
- [ ] Modules loaded
- [ ] Environment activated
- [ ] `cd` to project directory
- [ ] Dry run completed successfully
- [ ] Ready to run full experiment

**Then run:**
```bash
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn --epochs 500
```

Good luck! 🚀
