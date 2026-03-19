#!/bin/bash
# Setup script for fixing dependencies in allocation box
# Run this after: salloc --account=def-hsajjad --nodes=1 --gres=gpu:l40:1 --cpus-per-task=12 --mem=128000 --time=12:00:00

set -e  # Exit on error

echo "========================================="
echo "  Dependency Setup for Circuit Pruning"
echo "========================================="

# Load modules
echo ""
echo "Loading modules..."
module load StdEnv/2023 python-build-bundle/2025a gcc arrow

# Activate environment
echo ""
echo "Activating environment..."
source /home/dogar/project_env/bin/activate

# Show current versions
echo ""
echo "Current package versions:"
pip list | grep -E "(torch|flash|transformers)"

# Strategy: Upgrade to latest stable versions
# We'll use torch 2.10.0 (latest on CC) and rebuild flash-attn if needed

echo ""
echo "========================================="
echo "  Step 1: Uninstall conflicting packages"
echo "========================================="
pip uninstall -y flash-attn torchaudio

echo ""
echo "========================================="
echo "  Step 2: Install matching torch ecosystem"
echo "========================================="
# Install torch 2.10.0 + torchvision 0.25.0 (already compatible)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

echo ""
echo "========================================="
echo "  Step 3: Install/compile flash-attention"
echo "========================================="
# Try to install precompiled version first
echo "Attempting to install precompiled flash-attn..."
if pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "✓ Precompiled flash-attn installed successfully"
else
    echo "⚠ Precompiled version not available or incompatible"
    echo "Compiling flash-attn from source (this may take 10-30 minutes)..."

    # Ensure CUDA is available for compilation
    if ! nvidia-smi &>/dev/null; then
        echo "ERROR: No GPU detected. Make sure you're in an allocation with GPU access."
        exit 1
    fi

    # Install from source with proper CUDA setup
    pip install flash-attn --no-build-isolation
fi

echo ""
echo "========================================="
echo "  Step 4: Verify installation"
echo "========================================="
echo ""
echo "Final package versions:"
pip list | grep -E "(torch|flash|transformers)"

echo ""
echo "Testing imports..."
python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ CUDA version: {torch.version.cuda}')

import torchvision
print(f'✓ Torchvision: {torchvision.__version__}')

from transformers import AutoTokenizer, LlamaForCausalLM
print('✓ Transformers imports successful')

try:
    import flash_attn
    print(f'✓ Flash Attention: {flash_attn.__version__}')
    FLASH_AVAILABLE = True
except Exception as e:
    print(f'⚠ Flash Attention not available: {e}')
    FLASH_AVAILABLE = False
"

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "You can now run your script with:"
echo "  python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn"
echo ""
echo "If flash-attn failed to install, run without --flash-attn flag:"
echo "  python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95"
echo ""
