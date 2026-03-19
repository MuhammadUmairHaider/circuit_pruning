#!/usr/bin/env python3
"""
Quick verification script to check if environment is ready.
Run this in your allocation before starting the main experiment.
"""

import sys

def check_package(name, import_name=None):
    """Check if a package can be imported."""
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name:20s} {version}")
        return True
    except Exception as e:
        print(f"✗ {name:20s} FAILED: {e}")
        return False

def main():
    print("="*60)
    print("  Environment Verification")
    print("="*60)
    print()

    # Core packages
    print("Core Packages:")
    all_ok = True
    all_ok &= check_package("torch")
    all_ok &= check_package("torchvision")
    all_ok &= check_package("transformers")
    print()

    # Optional packages
    print("Optional Packages:")
    flash_ok = check_package("flash-attn", "flash_attn")
    print()

    # Check CUDA
    print("CUDA Setup:")
    try:
        import torch
        print(f"✓ CUDA available:      {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version:        {torch.version.cuda}")
            print(f"✓ GPU count:           {torch.cuda.device_count()}")
            print(f"✓ Current GPU:         {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - make sure you're in GPU allocation")
            all_ok = False
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        all_ok = False
    print()

    # Check transformers can load LLaMA
    print("Model Loading Test:")
    try:
        from transformers import AutoTokenizer, LlamaForCausalLM
        print("✓ LlamaForCausalLM     import successful")
    except Exception as e:
        print(f"✗ LlamaForCausalLM     FAILED: {e}")
        all_ok = False
    print()

    # Version requirements
    print("Version Checks:")
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # Remove +computecanada
        major, minor = map(int, torch_version.split('.')[:2])
        if major == 2 and minor >= 10:
            print(f"✓ PyTorch version:     {torch.__version__} (>= 2.10.0)")
        else:
            print(f"⚠ PyTorch version:     {torch.__version__} (expected >= 2.10.0)")
    except:
        pass

    print()
    print("="*60)

    if all_ok:
        print("✅ All required packages are working!")
        if flash_ok:
            print("✅ Flash Attention is available - use --flash-attn flag")
        else:
            print("⚠  Flash Attention not available - run WITHOUT --flash-attn flag")
        print()
        print("You're ready to run:")
        if flash_ok:
            print("  python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn --dry-run")
        else:
            print("  python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --dry-run")
        return 0
    else:
        print("❌ Some packages are missing or broken")
        print()
        print("Try fixing with:")
        print("  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0")
        print("  pip install flash-attn --no-build-isolation  # Optional, for speedup")
        return 1

if __name__ == "__main__":
    sys.exit(main())