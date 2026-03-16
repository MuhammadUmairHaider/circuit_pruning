# Which Training Script Should I Use?

## Quick Decision Tree

```
Do you know exactly what accuracy you want?
│
├─ YES → Use ioi_llama_hybrid_adaptive.py (RECOMMENDED)
│        python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn
│
└─ NO → Do you have a preference?
        │
        ├─ I want maximum sparsity → Use ioi_llama_fully_adaptive.py --aggressive
        │
        ├─ I want safe/conservative → Use ioi_llama_fully_adaptive.py --conservative
        │
        └─ Let it decide → Use ioi_llama_fully_adaptive.py --flash-attn
```

---

## Three Versions Explained

### 1. **ioi_llama_hybrid_adaptive.py** ⭐ RECOMMENDED

**Best for**: Most users who have a target accuracy in mind

**What you specify**: Target accuracy only (e.g., "I want 95% of baseline accuracy")

**What it discovers**: Maximum sparsity achievable at that accuracy

**Example**:
```bash
# I want 95% of baseline accuracy, auto-discover sparsity
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn

# Or fully automatic (no targets)
python ioi_llama_hybrid_adaptive.py --fully-adaptive --flash-attn
```

**Pros**:
- ✅ Most intuitive (you control accuracy, it optimizes sparsity)
- ✅ Flexible (can run fully adaptive too)
- ✅ Best of both worlds

**Cons**:
- ❌ Need to know roughly what accuracy you want

---

### 2. **ioi_llama_fully_adaptive.py** 🔄 ZERO CONFIG

**Best for**: Exploratory work, don't know what to expect

**What you specify**: Nothing! (optional: conservative/aggressive mode)

**What it discovers**: Optimal accuracy/sparsity tradeoff automatically

**Example**:
```bash
# Fully automatic - zero config
python ioi_llama_fully_adaptive.py --flash-attn

# Conservative (min 90% baseline accuracy)
python ioi_llama_fully_adaptive.py --conservative --flash-attn

# Aggressive (min 80% baseline accuracy, max sparsity)
python ioi_llama_fully_adaptive.py --aggressive --flash-attn
```

**Pros**:
- ✅ Zero configuration needed
- ✅ Great for exploration
- ✅ Automatically finds Pareto frontier

**Cons**:
- ❌ Less control over final accuracy
- ❌ May be too conservative or too aggressive for your needs

---

### 3. **ioi_llama_adaptive.py** 🎯 DUAL TARGETS

**Best for**: When you know both desired accuracy AND sparsity

**What you specify**: Both target accuracy AND target sparsity

**What it discovers**: How to get there

**Example**:
```bash
python ioi_llama_adaptive.py \
    --target-accuracy 0.95 \
    --target-sparsity 0.8 \
    --flash-attn
```

**Pros**:
- ✅ Most control
- ✅ Good if you know exactly what you want

**Cons**:
- ❌ Targets might be incompatible (can't achieve both)
- ❌ More configuration needed
- ❌ Less flexible than hybrid

---

## Detailed Comparison

| Feature | Hybrid ⭐ | Fully Adaptive 🔄 | Dual Targets 🎯 |
|---------|----------|-------------------|-----------------|
| **Config needed** | Target accuracy OR none | None | Both targets |
| **Flexibility** | High | Very High | Medium |
| **Control** | Medium | Low | High |
| **Best for** | Most users | Exploration | Specific goals |
| **Learning curve** | Easy | Easiest | Medium |
| **Risk of failure** | Low | Very Low | Medium |

---

## Usage Examples

### Scenario 1: "I want to maintain high accuracy"
```bash
# Use hybrid with high target accuracy
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.98 --flash-attn
```

### Scenario 2: "I want maximum compression"
```bash
# Use fully adaptive in aggressive mode
python ioi_llama_fully_adaptive.py --aggressive --flash-attn
```

### Scenario 3: "I don't know what's possible"
```bash
# Use fully adaptive in default mode
python ioi_llama_fully_adaptive.py --flash-attn

# Check the output, then run hybrid with specific target if needed
```

### Scenario 4: "I need exactly 95% accuracy and 80% sparsity"
```bash
# Use dual targets (but be prepared - might not be achievable!)
python ioi_llama_adaptive.py \
    --target-accuracy 0.95 \
    --target-sparsity 0.8 \
    --flash-attn
```

### Scenario 5: "Quick test to see what's possible"
```bash
# Dry run with fully adaptive
python ioi_llama_fully_adaptive.py --dry-run --flash-attn
```

---

## Recommendations by Use Case

### Research / Paper
**Use**: `ioi_llama_hybrid_adaptive.py`
- Run multiple experiments with different target accuracies
- Plot accuracy vs sparsity curve
- Report Pareto frontier

```bash
for acc in 0.90 0.92 0.94 0.96 0.98; do
    python ioi_llama_hybrid_adaptive.py --target-accuracy $acc --flash-attn
done
```

### Production / Deployment
**Use**: `ioi_llama_hybrid_adaptive.py` with specific accuracy target
- You know your accuracy requirements
- Maximize compression while meeting requirements

```bash
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn
```

### Exploration / Development
**Use**: `ioi_llama_fully_adaptive.py`
- Quick iteration
- Discover what's possible
- No prior knowledge needed

```bash
python ioi_llama_fully_adaptive.py --dry-run --flash-attn  # Quick test
python ioi_llama_fully_adaptive.py --flash-attn            # Full run
```

---

## Migration from Original

If you were using the original `ioi_llama.py`:

```bash
# Old way (manual lambda tuning)
python ioi_llama.py

# New way - equivalent but automatic
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn
```

**Advantages**:
- No more manual lambda tuning
- Faster (Flash Attention)
- Early stopping
- Automatic visualization

---

## Advanced: Combining Approaches

You can run different versions in sequence:

```bash
# Step 1: Explore with fully adaptive
python ioi_llama_fully_adaptive.py --flash-attn

# Output: "Discovered optimal at accuracy=0.92, sparsity=0.83"

# Step 2: Fine-tune with hybrid at specific accuracy
python ioi_llama_hybrid_adaptive.py --target-accuracy 0.92 --flash-attn

# Result: Maximum sparsity at exactly 92% accuracy
```

---

## Summary

**TLDR**:
- 🥇 **Most users**: `ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn`
- 🥈 **Explorers**: `ioi_llama_fully_adaptive.py --flash-attn`
- 🥉 **Specific goals**: `ioi_llama_adaptive.py --target-accuracy 0.95 --target-sparsity 0.8 --flash-attn`

**When in doubt**: Start with fully adaptive to see what's possible, then use hybrid with your preferred accuracy target.
