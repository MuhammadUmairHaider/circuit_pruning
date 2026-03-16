"""
Adaptive Pruning Scheduler for Circuit Discovery.

Automatically adjusts sparsity hyperparameters based on training dynamics
to make the pruning process smooth and reduce manual hyperparameter tuning.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AdaptiveSchedulerConfig:
    """Configuration for adaptive pruning scheduler."""

    # Target metrics
    target_accuracy: float = 0.95  # Target accuracy relative to baseline
    target_sparsity: float = 0.8   # Target fraction of gates to prune (0.8 = 80% pruned)

    # Warmup
    warmup_steps: int = 1000

    # Adaptation parameters
    sparsity_adaptation_rate: float = 0.1  # How aggressively to adjust lambdas
    accuracy_tolerance: float = 0.05       # Accuracy drop tolerance

    # Lambda bounds (prevent extreme values)
    min_lambda: float = 1e-4
    max_lambda: float = 10.0

    # Moving average for smoothing
    ema_alpha: float = 0.9  # Exponential moving average coefficient


class AdaptivePruningScheduler:
    """
    Dynamically adjusts sparsity loss weights (lambdas) based on:
    1. Current accuracy vs baseline
    2. Current sparsity vs target
    3. Training stability (gradient magnitudes, loss trends)

    Goal: Maintain accuracy while gradually increasing sparsity.
    """

    def __init__(self, config: AdaptiveSchedulerConfig, baseline_accuracy: float):
        self.config = config
        self.baseline_accuracy = baseline_accuracy
        self.target_accuracy = baseline_accuracy * config.target_accuracy

        # State tracking
        self.step = 0
        self.ema_accuracy = baseline_accuracy
        self.ema_sparsity = 0.0
        self.ema_kl_loss = 0.0

        # Lambda adjustments (multiplicative factors)
        self.lambda_multipliers = {
            'attention_heads': 1.0,
            'attention_neurons': 1.0,
            'mlp_hidden': 1.0,
            'mlp_output': 1.0,
            'attention_blocks': 1.0,
            'mlp_blocks': 1.0,
            'full_layers': 1.0,
            'embedding': 1.0,
        }

        # History for analysis
        self.history = {
            'step': [],
            'accuracy': [],
            'sparsity': [],
            'kl_loss': [],
            'lambda_mults': [],
        }

    def step_update(
        self,
        step: int,
        accuracy: float,
        sparsity_rate: float,
        kl_loss: float,
        gate_stats: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Update scheduler state and return new lambda multipliers.

        Args:
            step: Current training step
            accuracy: Current validation accuracy
            sparsity_rate: Current fraction of pruned gates (0-1)
            kl_loss: Current KL divergence loss
            gate_stats: Optional dict with per-component gate statistics

        Returns:
            Dict of lambda multipliers to apply to base lambdas
        """
        self.step = step

        # Update EMAs for smooth tracking
        alpha = self.config.ema_alpha
        self.ema_accuracy = alpha * self.ema_accuracy + (1 - alpha) * accuracy
        self.ema_sparsity = alpha * self.ema_sparsity + (1 - alpha) * sparsity_rate
        self.ema_kl_loss = alpha * self.ema_kl_loss + (1 - alpha) * kl_loss

        # Record history
        self.history['step'].append(step)
        self.history['accuracy'].append(accuracy)
        self.history['sparsity'].append(sparsity_rate)
        self.history['kl_loss'].append(kl_loss)
        self.history['lambda_mults'].append(dict(self.lambda_multipliers))

        # Don't adjust during warmup
        if step < self.config.warmup_steps:
            return self._get_warmup_multipliers(step)

        # Compute adjustment signals
        accuracy_gap = self.target_accuracy - self.ema_accuracy
        sparsity_gap = self.config.target_sparsity - self.ema_sparsity

        # Adaptive strategy:
        # 1. If accuracy is good and sparsity too low -> increase pruning pressure
        # 2. If accuracy is bad -> decrease pruning pressure
        # 3. If accuracy is good and sparsity on target -> maintain

        if accuracy_gap < -self.config.accuracy_tolerance:
            # Accuracy too low - reduce pruning pressure
            adjustment = -self.config.sparsity_adaptation_rate
            reason = "accuracy_recovery"
        elif accuracy_gap > self.config.accuracy_tolerance and sparsity_gap > 0.1:
            # Accuracy good, sparsity too low - increase pruning
            adjustment = self.config.sparsity_adaptation_rate
            reason = "increase_sparsity"
        else:
            # On target - fine-tune
            adjustment = sparsity_gap * self.config.sparsity_adaptation_rate * 0.1
            reason = "fine_tune"

        # Apply adjustment to all lambda multipliers
        for key in self.lambda_multipliers:
            self.lambda_multipliers[key] *= (1 + adjustment)
            # Clip to bounds
            self.lambda_multipliers[key] = np.clip(
                self.lambda_multipliers[key],
                self.config.min_lambda,
                self.config.max_lambda
            )

        if step % 100 == 0:
            print(f"\n[Scheduler] Step {step}: {reason}")
            print(f"  Accuracy: {accuracy:.4f} (target: {self.target_accuracy:.4f}, gap: {accuracy_gap:+.4f})")
            print(f"  Sparsity: {sparsity_rate:.4f} (target: {self.config.target_sparsity:.4f}, gap: {sparsity_gap:+.4f})")
            print(f"  Lambda mult: {self.lambda_multipliers['attention_heads']:.3f}")

        return dict(self.lambda_multipliers)

    def _get_warmup_multipliers(self, step: int) -> Dict[str, float]:
        """Gradual warmup to avoid early aggressive pruning."""
        warmup_fraction = step / self.config.warmup_steps
        # Exponential warmup: 0.01 -> 1.0
        warmup_scale = 0.01 * (100 ** warmup_fraction)

        return {key: warmup_scale for key in self.lambda_multipliers}

    def get_adjusted_lambdas(self, base_lambdas: Dict[str, float]) -> Dict[str, float]:
        """Apply current multipliers to base lambda values."""
        adjusted = {}
        for key, base_val in base_lambdas.items():
            mult = self.lambda_multipliers.get(key, 1.0)
            adjusted[key] = base_val * mult
        return adjusted

    def should_stop_early(self, patience: int = 50) -> bool:
        """Check if training has converged and can stop early."""
        if len(self.history['sparsity']) < patience:
            return False

        recent_sparsity = self.history['sparsity'][-patience:]
        recent_accuracy = self.history['accuracy'][-patience:]

        # Check if sparsity has plateaued
        sparsity_std = np.std(recent_sparsity)
        sparsity_on_target = abs(self.ema_sparsity - self.config.target_sparsity) < 0.05

        # Check if accuracy is stable and acceptable
        accuracy_stable = np.std(recent_accuracy) < 0.01
        accuracy_acceptable = self.ema_accuracy >= self.target_accuracy - self.config.accuracy_tolerance

        converged = (sparsity_std < 0.01 and sparsity_on_target and
                    accuracy_stable and accuracy_acceptable)

        if converged:
            print(f"\n[Scheduler] Convergence detected!")
            print(f"  Final sparsity: {self.ema_sparsity:.4f} (target: {self.config.target_sparsity:.4f})")
            print(f"  Final accuracy: {self.ema_accuracy:.4f} (target: {self.target_accuracy:.4f})")

        return converged

    def plot_training_dynamics(self, save_path: str = "training_dynamics.png"):
        """Plot the training history for analysis."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        steps = self.history['step']

        # Accuracy
        axes[0, 0].plot(steps, self.history['accuracy'], alpha=0.5, label='Accuracy')
        axes[0, 0].axhline(self.target_accuracy, color='r', linestyle='--', label='Target')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Sparsity
        axes[0, 1].plot(steps, self.history['sparsity'], alpha=0.5, label='Sparsity')
        axes[0, 1].axhline(self.config.target_sparsity, color='r', linestyle='--', label='Target')
        axes[0, 1].set_ylabel('Sparsity Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # KL Loss
        axes[1, 0].plot(steps, self.history['kl_loss'], alpha=0.5)
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')

        # Lambda multipliers
        if self.history['lambda_mults']:
            key = 'attention_heads'  # Representative key
            mults = [lm[key] for lm in self.history['lambda_mults']]
            axes[1, 1].plot(steps, mults, alpha=0.5)
            axes[1, 1].set_ylabel('Lambda Multiplier')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training dynamics plot saved to {save_path}")


class ProgressiveSparsityScheduler:
    """
    Simpler alternative: Gradually increase sparsity target over training.

    Phase 1 (0-30%): Learn task with minimal pruning
    Phase 2 (30-70%): Gradual sparsity increase
    Phase 3 (70-100%): Aggressive pruning to target
    """

    def __init__(
        self,
        total_steps: int,
        target_sparsity: float = 0.8,
        warmup_fraction: float = 0.3,
        aggressive_fraction: float = 0.7
    ):
        self.total_steps = total_steps
        self.target_sparsity = target_sparsity
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.aggressive_steps = int(total_steps * aggressive_fraction)

    def get_sparsity_weight(self, step: int) -> float:
        """Return sparsity loss weight for current step."""
        if step < self.warmup_steps:
            # Phase 1: Minimal pruning
            return 0.01
        elif step < self.aggressive_steps:
            # Phase 2: Linear ramp
            progress = (step - self.warmup_steps) / (self.aggressive_steps - self.warmup_steps)
            return 0.01 + progress * 0.99  # 0.01 -> 1.0
        else:
            # Phase 3: Aggressive pruning
            progress = (step - self.aggressive_steps) / (self.total_steps - self.aggressive_steps)
            return 1.0 + progress * 2.0  # 1.0 -> 3.0
