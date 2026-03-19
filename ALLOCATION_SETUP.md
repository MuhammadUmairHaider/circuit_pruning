# Allocation Box Setup Guide

## Quick Start

### 1. Request Allocation
```bash
salloc --account=def-hsajjad --nodes=1 --gres=gpu:l40:1 --cpus-per-task=12 --mem=128000 --time=12:00:00
```

### 2. Load Modules
```bash
module load StdEnv/2023 python-build-bundle/2025a gcc arrow
```

### 3. Activate Environment
```bash
source /home/dogar/project_env/bin/activate
```

### 4. Fix Dependencies

**Option A: Use the automated script (Recommended)**
```bash
cd /gpfs/project/6075961/dogar/circuit_latest/circuit_pruning
bash setup_dependencies.sh
```

**Option B: Manual commands**
```bash
# Uninstall conflicting packages
pip uninstall -y flash-attn torchaudio

# Install torch 2.10.0 ecosystem
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# Try to install flash-attention (precompiled for torch 2.10)
pip install flash-attn --no-build-isolation

# If above fails, compile from source (takes ~15-20 minutes):
pip install flash-attn --no-build-isolation --force-reinstall

# Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
from transformers import LlamaForCausalLM
print('Transformers: OK')
try:
    import flash_attn
    print(f'Flash-Attn: {flash_attn.__version__}')
except:
    print('Flash-Attn: NOT AVAILABLE')
"
```

### 5. Run Your Experiment

**With Flash Attention (faster, recommended):**
```bash
cd /gpfs/project/6075961/dogar/circuit_latest/circuit_pruning
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn --epochs 500
```

**Without Flash Attention (if installation failed):**
```bash
cd /gpfs/project/6075961/dogar/circuit_latest/circuit_pruning
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --epochs 500
```

---

## Dependency Resolution Strategy

### Current Issue
- `flash-attn 2.8.3` is compiled for `torch 2.9.1`
- `torchvision 0.25.0` requires `torch 2.10.0`
- Mismatched versions cause import errors

### Solution
Upgrade entire PyTorch ecosystem to 2.10.0:
- `torch 2.10.0` (latest stable on CC)
- `torchvision 0.25.0` (matches torch 2.10)
- `torchaudio 2.10.0` (matches torch 2.10)
- `flash-attn` - recompile/reinstall for torch 2.10

### Why This Works
- Compute Canada provides precompiled wheels for torch 2.10.0
- Flash-attention can be compiled from source if needed
- All packages will be on compatible versions

---

## Troubleshooting

### Flash Attention Compilation Fails
If flash-attention fails to compile, it's not critical. Just run without `--flash-attn`:
- You'll still get correct results
- Training will be ~2x slower
- But it's much simpler and guaranteed to work

### CUDA Not Available
Make sure you're in the allocation with GPU:
```bash
nvidia-smi  # Should show L40 GPU
```

### Import Errors Persist
Check versions:
```bash
pip list | grep -E "(torch|flash|transformers)"
```

All torch packages should be on 2.10.0 or compatible versions.

---

## Expected Package Versions After Setup

```
torch                   2.10.0
torchvision             0.25.0
torchaudio              2.10.0
transformers            5.2.0
flash-attn              2.x.x (latest compatible with torch 2.10)
```

---

## Quick Test Before Long Run

Always do a dry run first:
```bash
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn --dry-run
```

This runs 2 epochs to verify everything works before committing to a 12-hour run.
