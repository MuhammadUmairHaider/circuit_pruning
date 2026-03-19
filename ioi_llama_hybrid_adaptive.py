"""
IOI Circuit Discovery with Hybrid Adaptive Pruning.

Flexible configuration:
1. Specify target accuracy, auto-discover optimal sparsity
2. Or run fully adaptive (no targets needed)

This is the recommended version for most users!

Usage:
    # Auto-discover sparsity, maintain 95% baseline accuracy
    python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95

    # Fully automatic (both accuracy and sparsity adaptive)
    python ioi_llama_hybrid_adaptive.py --fully-adaptive

    # With speedups
    python ioi_llama_hybrid_adaptive.py --target-accuracy 0.95 --flash-attn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import time
import argparse
import os
import numpy as np
from collections import deque

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from dataset.ioi_llama import (
    IOIDatasetLlama,
    generate_ioi_data_llama,
    run_evaluation,
    filter_dataset_by_model_correctness,
)
from utils import disable_dropout, analyze_and_finalize_circuit
from dataclasses import dataclass


# ==============================================================================
# HYBRID ADAPTIVE SCHEDULER
# ==============================================================================

@dataclass
class HybridAdaptiveConfig:
    """Configuration for hybrid adaptive scheduler."""
    warmup_steps: int = 10

    # Target accuracy (None = fully adaptive)
    target_accuracy: Optional[float] = None  # e.g., 0.95 = 95% of baseline

    # Constraints for fully adaptive mode
    min_accuracy_fraction: float = 0.85
    accuracy_tolerance: float = 0.02

    # Adaptation parameters
    lambda_adjustment_rate: float = 0.15
    window_size: int = 20

    # Convergence
    convergence_patience: int = 50
    min_training_epochs: int = 100

    # Lambda bounds
    min_lambda: float = 1e-4
    max_lambda: float = 20.0

    # EMA smoothing
    ema_alpha: float = 0.85


class HybridAdaptiveScheduler:
    """
    Hybrid scheduler that can work in two modes:

    1. Target Accuracy Mode (recommended):
       - You specify target_accuracy (e.g., 0.95 = 95% of baseline)
       - Scheduler maximizes sparsity while maintaining that accuracy

    2. Fully Adaptive Mode:
       - No targets needed
       - Automatically finds best accuracy/sparsity tradeoff
    """

    def __init__(self, config: HybridAdaptiveConfig, baseline_accuracy: float):
        self.config = config
        self.baseline_accuracy = baseline_accuracy

        # Determine mode and thresholds
        if config.target_accuracy is not None:
            self.mode = "target_accuracy"
            self.target_accuracy = baseline_accuracy * config.target_accuracy
            self.min_acceptable_accuracy = self.target_accuracy - config.accuracy_tolerance
            print(f"\n🎯 Mode: TARGET ACCURACY")
            print(f"   Target: {self.target_accuracy:.4f} ({config.target_accuracy*100:.1f}% of baseline)")
            print(f"   Will maximize sparsity while maintaining this accuracy")
        else:
            self.mode = "fully_adaptive"
            self.target_accuracy = None
            self.min_acceptable_accuracy = baseline_accuracy * config.min_accuracy_fraction
            print(f"\n🔄 Mode: FULLY ADAPTIVE")
            print(f"   Will automatically find optimal accuracy/sparsity tradeoff")
            print(f"   Minimum acceptable: {self.min_acceptable_accuracy:.4f}")

        # State
        self.step = 0
        self.epoch = 0

        # EMA metrics
        self.ema_accuracy = baseline_accuracy
        self.ema_sparsity = 0.0
        self.ema_kl_loss = 0.0

        # Windows for trend detection
        self.accuracy_window = deque(maxlen=config.window_size)
        self.sparsity_window = deque(maxlen=config.window_size)

        # Lambda state
        self.lambda_multiplier = 0.1  # Start conservative
        self.best_lambda = 0.1

        # Best tracking
        self.best_sparsity = 0.0
        self.best_accuracy = baseline_accuracy
        self.epochs_since_improvement = 0

        # Phase tracking
        self.phase = "warmup"

        # History
        self.history = {
            'step': [], 'epoch': [], 'accuracy': [], 'sparsity': [],
            'kl_loss': [], 'lambda_mult': [], 'phase': [], 'action': [],
        }

    def step_update(
        self,
        step: int,
        epoch: int,
        accuracy: float,
        sparsity_rate: float,
        kl_loss: float,
    ) -> Dict[str, float]:
        """Update and return lambda multiplier."""
        self.step = step
        self.epoch = epoch

        # Update EMAs
        alpha = self.config.ema_alpha
        self.ema_accuracy = alpha * self.ema_accuracy + (1 - alpha) * accuracy
        self.ema_sparsity = alpha * self.ema_sparsity + (1 - alpha) * sparsity_rate
        self.ema_kl_loss = alpha * self.ema_kl_loss + (1 - alpha) * kl_loss

        # Update windows
        self.accuracy_window.append(accuracy)
        self.sparsity_window.append(sparsity_rate)

        # Determine action
        if self.mode == "target_accuracy":
            action = self._determine_action_target_mode()
        else:
            action = self._determine_action_adaptive_mode()

        # Adjust lambda
        old_lambda = self.lambda_multiplier
        self.lambda_multiplier = self._adjust_lambda(action)

        # Record
        self.history['step'].append(step)
        self.history['epoch'].append(epoch)
        self.history['accuracy'].append(accuracy)
        self.history['sparsity'].append(sparsity_rate)
        self.history['kl_loss'].append(kl_loss)
        self.history['lambda_mult'].append(self.lambda_multiplier)
        self.history['phase'].append(self.phase)
        self.history['action'].append(action)

        # Logging
        if epoch % 10 == 0:
            self._log_status(action, old_lambda)

        return {'multiplier': self.lambda_multiplier}

    def _determine_action_target_mode(self) -> str:
        """
        Target accuracy mode: Maximize sparsity while maintaining target accuracy.

        Strategy:
        - If accuracy > target: Increase pruning (push for more sparsity)
        - If accuracy < target: Decrease pruning (recover accuracy)
        - If accuracy ≈ target: Fine-tune (explore/maintain)
        """
        # Warmup
        if self.step < self.config.warmup_steps:
            self.phase = "warmup"
            return "warmup"

        if len(self.accuracy_window) < 3:
            return "maintain"

        # Calculate trend
        acc_trend = self._calculate_trend(self.accuracy_window)

        # Accuracy relative to target
        acc_gap = self.ema_accuracy - self.target_accuracy
        acc_dropping = acc_trend < -0.005

        # Update best sparsity
        if self.ema_accuracy >= self.min_acceptable_accuracy:
            if self.ema_sparsity > self.best_sparsity:
                self.best_sparsity = self.ema_sparsity
                self.best_lambda = self.lambda_multiplier
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1

        # Decision logic
        if self.ema_accuracy < self.min_acceptable_accuracy:
            # Below minimum - emergency recovery
            self.phase = "recovery"
            return "decrease_aggressive"

        elif acc_gap < -self.config.accuracy_tolerance:
            # Below target - decrease pruning
            self.phase = "recovery"
            return "decrease"

        elif acc_gap > 2 * self.config.accuracy_tolerance:
            # Well above target - can prune more aggressively
            self.phase = "exploration"
            return "increase"

        elif acc_dropping and abs(acc_gap) < self.config.accuracy_tolerance:
            # At target but dropping - careful decrease
            self.phase = "fine_tuning"
            return "decrease"

        else:
            # Near target and stable - fine-tune
            self.phase = "fine_tuning"
            # Small random exploration
            if np.random.random() < 0.2:
                return "explore"
            else:
                return "maintain"

    def _determine_action_adaptive_mode(self) -> str:
        """
        Fully adaptive mode: Find optimal tradeoff automatically.

        Similar to fully adaptive scheduler in v2.
        """
        if self.step < self.config.warmup_steps:
            self.phase = "warmup"
            return "warmup"

        if len(self.accuracy_window) < 3:
            return "maintain"

        acc_trend = self._calculate_trend(self.accuracy_window)
        sparsity_trend = self._calculate_trend(self.sparsity_window)

        acc_ok = self.ema_accuracy >= self.min_acceptable_accuracy + self.config.accuracy_tolerance
        acc_dropping = acc_trend < -0.005
        sparsity_stuck = abs(sparsity_trend) < 0.001

        # Update best
        if acc_ok and self.ema_sparsity > self.best_sparsity:
            self.best_sparsity = self.ema_sparsity
            self.best_lambda = self.lambda_multiplier
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # Decision
        if not acc_ok:
            self.phase = "recovery"
            return "decrease_aggressive"
        elif acc_dropping and self.ema_sparsity > 0.3:
            self.phase = "exploitation"
            return "decrease"
        elif sparsity_stuck and acc_ok:
            self.phase = "exploration"
            return "increase"
        else:
            self.phase = "fine_tuning"
            if np.random.random() < 0.3:
                return "explore"
            else:
                return "maintain"

    def _adjust_lambda(self, action: str) -> float:
        """Adjust lambda based on action."""
        rate = self.config.lambda_adjustment_rate

        adjustments = {
            'warmup': lambda l: l * 1.05,
            'increase': lambda l: l * (1 + rate),
            'decrease': lambda l: l * (1 - rate),
            'decrease_aggressive': lambda l: l * (1 - 2 * rate),
            'explore': lambda l: l * (1 + rate * 0.5),
            'maintain': lambda l: l,
        }

        new_lambda = adjustments.get(action, lambda l: l)(self.lambda_multiplier)
        return np.clip(new_lambda, self.config.min_lambda, self.config.max_lambda)

    def _calculate_trend(self, window: deque) -> float:
        """Calculate trend (slope) of recent values."""
        if len(window) < 3:
            return 0.0
        values = list(window)
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def should_stop_early(self) -> bool:
        """Check convergence."""
        if self.epoch < self.config.min_training_epochs:
            return False

        if self.epochs_since_improvement > self.config.convergence_patience:
            return True

        if len(self.sparsity_window) >= self.config.window_size:
            sparsity_stable = np.std(list(self.sparsity_window)) < 0.01
            acc_stable = np.std(list(self.accuracy_window)) < 0.01

            if sparsity_stable and acc_stable:
                if self.mode == "target_accuracy":
                    # Check if at target
                    return abs(self.ema_accuracy - self.target_accuracy) < self.config.accuracy_tolerance
                else:
                    # Check if at good tradeoff
                    return self.ema_accuracy >= self.min_acceptable_accuracy and self.ema_sparsity > 0.5

        return False

    def get_final_summary(self) -> Dict:
        """Get final summary."""
        return {
            'mode': self.mode,
            'baseline_accuracy': self.baseline_accuracy,
            'target_accuracy': self.target_accuracy,
            'final_accuracy': self.ema_accuracy,
            'final_sparsity': self.ema_sparsity,
            'best_sparsity': self.best_sparsity,
            'best_lambda': self.best_lambda,
            'total_epochs': self.epoch,
            'phase': self.phase,
            'accuracy_drop_pct': (self.baseline_accuracy - self.ema_accuracy) / self.baseline_accuracy * 100,
        }

    def _log_status(self, action: str, old_lambda: float):
        """Print status."""
        acc_drop_pct = (self.baseline_accuracy - self.ema_accuracy) / self.baseline_accuracy * 100

        print(f"\n[Scheduler] Epoch {self.epoch} | Phase: {self.phase} | Mode: {self.mode}")
        print(f"  Accuracy: {self.ema_accuracy:.4f} (baseline: {self.baseline_accuracy:.4f}, drop: {acc_drop_pct:.1f}%)")

        if self.target_accuracy:
            gap = self.ema_accuracy - self.target_accuracy
            print(f"  Target:   {self.target_accuracy:.4f} (gap: {gap:+.4f})")

        print(f"  Sparsity: {self.ema_sparsity:.4f} (best: {self.best_sparsity:.4f})")
        print(f"  KL Loss: {self.ema_kl_loss:.4f}")
        print(f"  Lambda:   {old_lambda:.3f} → {self.lambda_multiplier:.3f} (action: {action})")

    def plot_training_dynamics(self, save_path: str):
        """Plot training dynamics."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = self.history['epoch']

        # Accuracy
        ax = axes[0, 0]
        ax.plot(epochs, self.history['accuracy'], alpha=0.6, label='Accuracy')
        ax.axhline(self.baseline_accuracy, color='green', linestyle='--', label='Baseline', alpha=0.7)
        if self.target_accuracy:
            ax.axhline(self.target_accuracy, color='blue', linestyle='--', label='Target', alpha=0.7)
        ax.axhline(self.min_acceptable_accuracy, color='red', linestyle='--', label='Min acceptable', alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Accuracy ({self.mode})')

        # Sparsity
        ax = axes[0, 1]
        ax.plot(epochs, self.history['sparsity'], alpha=0.6, color='orange')
        if self.best_sparsity > 0:
            ax.axhline(self.best_sparsity, color='green', linestyle='--', label=f'Best: {self.best_sparsity:.3f}')
        ax.set_ylabel('Sparsity')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Sparsity Progression')

        # Lambda
        ax = axes[1, 0]
        ax.plot(epochs, self.history['lambda_mult'], alpha=0.6)
        ax.set_ylabel('Lambda Multiplier')
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title('Lambda Adaptation')

        # Tradeoff
        ax = axes[1, 1]
        scatter = ax.scatter(self.history['sparsity'], self.history['accuracy'],
                            c=epochs, cmap='viridis', alpha=0.6, s=20)
        if self.target_accuracy:
            ax.axhline(self.target_accuracy, color='blue', linestyle='--', alpha=0.5, label='Target')
        ax.axhline(self.min_acceptable_accuracy, color='red', linestyle='--', alpha=0.5, label='Min acceptable')

        # Mark best
        best_idx = np.argmax(self.history['sparsity'])
        ax.scatter(self.history['sparsity'][best_idx], self.history['accuracy'][best_idx],
                  color='red', s=200, marker='*', edgecolor='black', linewidth=2, zorder=10)

        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Accuracy-Sparsity Tradeoff')
        plt.colorbar(scatter, ax=ax, label='Epoch')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training plot saved: {save_path}")


# ==============================================================================
# Helper Functions
# ==============================================================================

@dataclass
class HybridLlamaPruningConfig(PruningConfig):
    """Base config for hybrid adaptive training."""
    init_value: float = 0.5
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.0

    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0

    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0

    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0

    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0

    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 1.0

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 1.0

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0

    prune_embedding: bool = False
    lambda_embedding: float = 1.0


def compute_overall_sparsity(model) -> float:
    """Compute overall sparsity rate."""
    from models.l0 import HardConcreteGate

    total_gates = 0
    open_gates = 0

    for module in model.modules():
        if isinstance(module, HardConcreteGate):
            with torch.no_grad():
                gates = module()
                total_gates += gates.numel()
                open_gates += (gates > 0.5).sum().item()

    return 1.0 - (open_gates / total_gates) if total_gates > 0 else 0.0


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid Adaptive Circuit Discovery")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)  # Lower LR for more stable training
    parser.add_argument('--batch-size', type=int, default=32)  # Larger batch for more stable gradients
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints_llama_hybrid')

    # Key parameter: target accuracy (None = fully adaptive)
    parser.add_argument('--target-accuracy', type=float, default=0.95,
                        help='Target accuracy as fraction of baseline (e.g., 0.95 = 95%%). Use --fully-adaptive to disable.')
    parser.add_argument('--fully-adaptive', action='store_true',
                        help='Fully adaptive mode (no target accuracy)')

    # Speedups
    parser.add_argument('--flash-attn', action='store_true')

    args = parser.parse_args()

    # Read HF token
    hf_token = args.hf_token
    if hf_token is None:
        token_file = "hf_tokken.txt"
        if os.path.exists(token_file):
            with open(token_file) as f:
                hf_token = f.read().strip()

    # Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("  HYBRID ADAPTIVE CIRCUIT DISCOVERY")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Flash Attention: {args.flash_attn}")

    # Load models
    print("\n--- Loading models ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    pruning_config = HybridLlamaPruningConfig()
    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model, pruning_config, **model_kwargs
    ).to(DEVICE).eval()

    full_model = LlamaForCausalLM.from_pretrained(
        args.model, **model_kwargs
    ).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    disable_dropout(circuit_model)

    # Freeze base, unfreeze gates
    GATE_PATTERNS = ('_gates.', '_gate.', 'embedding_gate.', 'layer_gates.')
    for name, param in circuit_model.named_parameters():
        is_gate = any(p in name for p in GATE_PATTERNS)
        param.requires_grad = is_gate
        if is_gate:
            param.data = param.data.float()

    # Data
    NUM_TRAIN = 10 if args.dry_run else 200
    NUM_VAL = 5 if args.dry_run else 200
    NUM_TEST = 5 if args.dry_run else 1000

    train_data = generate_ioi_data_llama(NUM_TRAIN, tokenizer, seed=42)
    val_data = generate_ioi_data_llama(NUM_VAL, tokenizer, seed=123)
    test_data = generate_ioi_data_llama(NUM_TEST, tokenizer, seed=456)

    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, DEVICE, args.batch_size)
    test_data = filter_dataset_by_model_correctness(test_data, full_model, tokenizer, DEVICE, args.batch_size)

    train_dataset = IOIDatasetLlama(train_data, tokenizer)
    val_dataset = IOIDatasetLlama(val_data, tokenizer)
    test_dataset = IOIDatasetLlama(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)  # shuffle=False required for cached logits
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    # Baseline
    print("\n--- Baseline evaluation ---")
    baseline_results = run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline",
        full_model_for_faithfulness=None,
        dataloader=test_dataloader,
        device=DEVICE,
        verbose=True,
        tokenizer=tokenizer,
    )
    base_accuracy = baseline_results['accuracy']
    print(f"🎯 Baseline: {base_accuracy:.4f}")

    # Initialize scheduler
    scheduler_config = HybridAdaptiveConfig(
        warmup_steps=pruning_config.sparsity_warmup_steps,
        target_accuracy=None if args.fully_adaptive else args.target_accuracy,
    )
    scheduler = HybridAdaptiveScheduler(scheduler_config, base_accuracy)

    # Setup training
    optimizer = AdamW([p for p in circuit_model.parameters() if p.requires_grad], lr=args.lr)

    # Pre-cache full model
    print("\n🚀 Caching full model outputs...")
    cached_train_logits = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = full_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False)
            cached_train_logits[batch_idx] = out.logits.detach()

    # Training loop
    print(f"\n{'='*80}")
    print("  TRAINING")
    print(f"{'='*80}\n")

    total_steps = 0
    NUM_EPOCHS = 2 if args.dry_run else args.epochs

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        circuit_model.train()

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )

            # KL loss
            total_kl = 0
            for i in range(outputs.logits.size(0)):
                t_start = batch['T_Start'][i].item() - 1
                t_end = batch['T_End'][i].item() - 1

                # Get valid sequence length (before padding)
                valid_length = batch['attention_mask'][i].sum().item()

                # Don't compute KL on padding positions
                end_pos = min(t_end, valid_length)

                if t_start < end_pos:
                    kl = F.kl_div(
                        F.log_softmax(outputs.logits[i, t_start:end_pos, :].float(), dim=-1),
                        F.log_softmax(cached_train_logits[batch_idx][i, t_start:end_pos, :].float(), dim=-1),
                        reduction='sum', log_target=True,
                    )
                    total_kl += kl
            kl_loss = total_kl / outputs.logits.size(0)
            # print(f"KL Loss: {kl_loss.item():.4f}")

            # Task loss
            pos_good = batch['T_Start'] - 1
            token_good = batch['target_tokens'][:, 0]
            pos_bad = batch['D_Start'] - 1
            token_bad = batch['distractor_tokens'][:, 0]
            batch_indices = torch.arange(outputs.logits.size(0), device=DEVICE)

            logit_good = outputs.logits[batch_indices, pos_good, token_good].float()
            logit_bad = outputs.logits[batch_indices, pos_bad, token_bad].float()
            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()

            # Sparsity loss with adaptive multiplier
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            sparsity_loss = sparsity_loss * scheduler.lambda_multiplier

            loss = kl_loss * 1.0 + sparsity_loss# + task_loss
            loss.backward()
            optimizer.step()

            total_steps += 1

        # Validation
        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            val_results = run_evaluation(
                model_to_eval=circuit_model,
                model_name=f"Ep{epoch+1}",
                full_model_for_faithfulness=full_model,
                dataloader=val_dataloader,
                device=DEVICE,
                verbose=False,
                tokenizer=tokenizer,
            )

            current_sparsity = compute_overall_sparsity(circuit_model)

            scheduler.step_update(
                step=total_steps,
                epoch=epoch + 1,
                accuracy=val_results['accuracy'],
                sparsity_rate=current_sparsity,
                kl_loss=val_results['kl_div'],
            )

            if scheduler.should_stop_early():
                print("\n🎉 Converged! Stopping early.")
                break

    # Save plot
    scheduler.plot_training_dynamics(os.path.join(args.save_dir, 'training.png'))

    # Final eval
    print("\n--- Final Evaluation ---")
    circuit_model.eval()
    analyze_and_finalize_circuit(circuit_model)
    final_results = run_evaluation(
        model_to_eval=circuit_model,
        model_name="Final",
        full_model_for_faithfulness=full_model,
        dataloader=test_dataloader,
        device=DEVICE,
        verbose=True,
        tokenizer=tokenizer,
    )

    summary = scheduler.get_final_summary()
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"Mode: {summary['mode']}")
    print(f"Baseline: {summary['baseline_accuracy']:.4f}")
    if summary['target_accuracy']:
        print(f"Target:   {summary['target_accuracy']:.4f}")
    print(f"Final:    {summary['final_accuracy']:.4f} ({summary['accuracy_drop_pct']:.1f}% drop)")
    print(f"Sparsity: {summary['final_sparsity']:.4f}")
    print(f"Epochs:   {summary['total_epochs']}")
    print(f"{'='*80}\n")
