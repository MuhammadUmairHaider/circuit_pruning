"""
Fully Adaptive Pruning Scheduler for Circuit Discovery.

No manual targets needed! Automatically finds the optimal accuracy/sparsity tradeoff
by monitoring training dynamics and adjusting pruning pressure intelligently.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from collections import deque


@dataclass
class FullyAdaptiveConfig:
    """Configuration for fully adaptive pruning scheduler."""

    # Warmup
    warmup_steps: int = 1000

    # Accuracy constraints
    min_accuracy_fraction: float = 0.85  # Never drop below 85% of baseline
    accuracy_buffer: float = 0.02        # Keep 2% safety margin

    # Sparsity preferences
    prefer_aggressive: bool = True       # Try to prune as much as possible
    min_sparsity_target: float = 0.5     # Don't bother if less than 50% pruned

    # Adaptation dynamics
    lambda_adjustment_rate: float = 0.15  # How fast to adjust (0.1-0.2 recommended)
    window_size: int = 20                # Look at last N validation points

    # Convergence detection
    convergence_patience: int = 50       # Epochs without improvement
    min_training_epochs: int = 100       # Don't stop too early

    # Lambda bounds
    min_lambda: float = 1e-4
    max_lambda: float = 20.0

    # EMA smoothing
    ema_alpha: float = 0.85


class FullyAdaptivePruningScheduler:
    """
    Fully adaptive scheduler that automatically finds optimal pruning.

    Strategy:
    1. Start conservative (low pruning pressure)
    2. Gradually increase until accuracy starts dropping
    3. Back off slightly to maintain accuracy
    4. Fine-tune around the accuracy frontier
    5. Stop when no more improvement possible

    No manual targets needed!
    """

    def __init__(self, config: FullyAdaptiveConfig, baseline_accuracy: float):
        self.config = config
        self.baseline_accuracy = baseline_accuracy
        self.min_acceptable_accuracy = baseline_accuracy * config.min_accuracy_fraction

        # State tracking
        self.step = 0
        self.epoch = 0

        # Metrics history
        self.ema_accuracy = baseline_accuracy
        self.ema_sparsity = 0.0
        self.ema_kl_loss = 0.0

        # Sliding window for trend detection
        self.accuracy_window = deque(maxlen=config.window_size)
        self.sparsity_window = deque(maxlen=config.window_size)
        self.kl_window = deque(maxlen=config.window_size)

        # Lambda state
        self.lambda_multiplier = 0.1  # Start conservative
        self.best_lambda = 0.1

        # Best metrics tracking
        self.best_sparsity = 0.0
        self.best_sparsity_at_acceptable_acc = 0.0
        self.best_accuracy = baseline_accuracy
        self.epochs_since_improvement = 0

        # Training phase
        self.phase = "warmup"  # warmup -> exploration -> exploitation -> convergence

        # Full history
        self.history = {
            'step': [],
            'epoch': [],
            'accuracy': [],
            'sparsity': [],
            'kl_loss': [],
            'lambda_mult': [],
            'phase': [],
            'action': [],
        }

    def step_update(
        self,
        step: int,
        epoch: int,
        accuracy: float,
        sparsity_rate: float,
        kl_loss: float,
    ) -> Dict[str, float]:
        """
        Update scheduler state and return new lambda multiplier.

        Returns:
            Dict with 'multiplier' key for lambda scaling
        """
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
        self.kl_window.append(kl_loss)

        # Determine action based on current phase
        action = self._determine_action()

        # Update lambda multiplier
        old_lambda = self.lambda_multiplier
        self.lambda_multiplier = self._adjust_lambda(action)

        # Record history
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

    def _determine_action(self) -> str:
        """
        Determine what action to take based on current state.

        Actions:
        - 'warmup': Gradually ramp up
        - 'increase': Push for more sparsity
        - 'decrease': Back off, accuracy dropping
        - 'maintain': Stay at current level
        - 'explore': Try small increase
        - 'exploit': Fine-tune around frontier
        """

        # Phase 1: Warmup
        if self.step < self.config.warmup_steps:
            self.phase = "warmup"
            return "warmup"

        # Check if we have enough data
        if len(self.accuracy_window) < 3:
            return "maintain"

        # Calculate trends
        acc_trend = self._calculate_trend(self.accuracy_window)
        sparsity_trend = self._calculate_trend(self.sparsity_window)

        # Current state
        acc_ok = self.ema_accuracy >= self.min_acceptable_accuracy + self.config.accuracy_buffer
        acc_dropping = acc_trend < -0.005  # Accuracy decreasing
        sparsity_low = self.ema_sparsity < self.config.min_sparsity_target
        sparsity_stuck = abs(sparsity_trend) < 0.001  # Not moving

        # Update best metrics
        if acc_ok and self.ema_sparsity > self.best_sparsity_at_acceptable_acc:
            self.best_sparsity_at_acceptable_acc = self.ema_sparsity
            self.best_lambda = self.lambda_multiplier
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # Decision logic
        if not acc_ok:
            # Accuracy too low - emergency back-off
            self.phase = "recovery"
            return "decrease_aggressive"

        elif acc_dropping and self.ema_sparsity > 0.3:
            # Accuracy starting to drop, we're near the frontier
            self.phase = "exploitation"
            return "decrease"

        elif sparsity_low or (sparsity_stuck and acc_ok):
            # We can push harder for sparsity
            self.phase = "exploration"
            return "increase"

        elif self.epochs_since_improvement > self.config.convergence_patience:
            # No improvement for a while, we're likely at optimum
            self.phase = "convergence"
            return "maintain"

        else:
            # Fine-tuning around the frontier
            self.phase = "exploitation"
            # Small exploration: try to squeeze out more sparsity
            if np.random.random() < 0.3:  # 30% chance to explore
                return "explore"
            else:
                return "maintain"

    def _adjust_lambda(self, action: str) -> float:
        """Adjust lambda multiplier based on action."""
        rate = self.config.lambda_adjustment_rate

        adjustments = {
            'warmup': lambda l: l * 1.05,  # Slow exponential ramp
            'increase': lambda l: l * (1 + rate),
            'decrease': lambda l: l * (1 - rate),
            'decrease_aggressive': lambda l: l * (1 - 2 * rate),
            'explore': lambda l: l * (1 + rate * 0.5),  # Small increase
            'maintain': lambda l: l,
        }

        new_lambda = adjustments.get(action, lambda l: l)(self.lambda_multiplier)

        # Clip to bounds
        new_lambda = np.clip(new_lambda, self.config.min_lambda, self.config.max_lambda)

        return new_lambda

    def _calculate_trend(self, window: deque) -> float:
        """Calculate trend (slope) of recent values."""
        if len(window) < 3:
            return 0.0

        values = list(window)
        x = np.arange(len(values))
        # Simple linear regression
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def should_stop_early(self) -> bool:
        """Check if training has converged."""
        if self.epoch < self.config.min_training_epochs:
            return False

        # Stop if we've been in convergence phase for a while
        if self.phase == "convergence" and self.epochs_since_improvement > self.config.convergence_patience:
            return True

        # Stop if sparsity is very high and stable
        if len(self.sparsity_window) >= self.config.window_size:
            recent_sparsity = list(self.sparsity_window)
            sparsity_stable = np.std(recent_sparsity) < 0.01
            sparsity_high = self.ema_sparsity > 0.8

            if sparsity_stable and sparsity_high and self.ema_accuracy >= self.min_acceptable_accuracy:
                return True

        return False

    def get_final_summary(self) -> Dict:
        """Get summary of discovered circuit."""
        return {
            'baseline_accuracy': self.baseline_accuracy,
            'final_accuracy': self.ema_accuracy,
            'final_sparsity': self.ema_sparsity,
            'best_sparsity_at_acceptable_acc': self.best_sparsity_at_acceptable_acc,
            'best_lambda': self.best_lambda,
            'total_epochs': self.epoch,
            'phase': self.phase,
            'accuracy_drop': self.baseline_accuracy - self.ema_accuracy,
            'accuracy_drop_pct': (self.baseline_accuracy - self.ema_accuracy) / self.baseline_accuracy * 100,
        }

    def _log_status(self, action: str, old_lambda: float):
        """Print current status."""
        acc_drop = (self.baseline_accuracy - self.ema_accuracy) / self.baseline_accuracy * 100

        print(f"\n[Adaptive Scheduler] Epoch {self.epoch} | Phase: {self.phase}")
        print(f"  Accuracy: {self.ema_accuracy:.4f} (baseline: {self.baseline_accuracy:.4f}, drop: {acc_drop:.1f}%)")
        print(f"  Sparsity: {self.ema_sparsity:.4f} (best: {self.best_sparsity_at_acceptable_acc:.4f})")
        print(f"  KL Div:   {self.ema_kl_loss:.4f}")
        print(f"  Lambda:   {old_lambda:.3f} → {self.lambda_multiplier:.3f} (action: {action})")
        print(f"  Epochs since improvement: {self.epochs_since_improvement}")

    def plot_training_dynamics(self, save_path: str = "adaptive_training.png"):
        """Plot training dynamics with phase annotations."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            print("matplotlib not available, skipping plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = self.history['epoch']

        # Accuracy plot
        ax = axes[0, 0]
        ax.plot(epochs, self.history['accuracy'], alpha=0.6, label='Accuracy', color='blue')
        ax.axhline(self.min_acceptable_accuracy, color='red', linestyle='--',
                   label=f'Min acceptable ({self.min_acceptable_accuracy:.3f})', alpha=0.7)
        ax.axhline(self.baseline_accuracy, color='green', linestyle='--',
                   label=f'Baseline ({self.baseline_accuracy:.3f})', alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_title('Accuracy Trajectory')

        # Sparsity plot
        ax = axes[0, 1]
        ax.plot(epochs, self.history['sparsity'], alpha=0.6, label='Sparsity', color='orange')
        if self.best_sparsity_at_acceptable_acc > 0:
            ax.axhline(self.best_sparsity_at_acceptable_acc, color='green', linestyle='--',
                      label=f'Best ({self.best_sparsity_at_acceptable_acc:.3f})', alpha=0.7)
        ax.set_ylabel('Sparsity Rate')
        ax.set_xlabel('Epoch')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_title('Sparsity Progression')

        # Lambda multiplier plot with phase coloring
        ax = axes[1, 0]

        # Color by phase
        phase_colors = {
            'warmup': 'gray',
            'exploration': 'blue',
            'exploitation': 'orange',
            'recovery': 'red',
            'convergence': 'green'
        }

        for i in range(len(epochs) - 1):
            phase = self.history['phase'][i]
            color = phase_colors.get(phase, 'black')
            ax.plot(epochs[i:i+2], self.history['lambda_mult'][i:i+2],
                   color=color, alpha=0.6, linewidth=2)

        ax.set_ylabel('Lambda Multiplier')
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title('Lambda Adaptation (colored by phase)')

        # Add legend for phases
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=phase.title())
                          for phase, color in phase_colors.items()]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)

        # Accuracy vs Sparsity tradeoff
        ax = axes[1, 1]
        scatter = ax.scatter(self.history['sparsity'], self.history['accuracy'],
                            c=self.history['epoch'], cmap='viridis', alpha=0.6, s=20)
        ax.axhline(self.min_acceptable_accuracy, color='red', linestyle='--', alpha=0.5)

        # Mark best point
        best_idx = np.argmax([s if a >= self.min_acceptable_accuracy else 0
                             for s, a in zip(self.history['sparsity'], self.history['accuracy'])])
        ax.scatter(self.history['sparsity'][best_idx], self.history['accuracy'][best_idx],
                  color='red', s=200, marker='*', edgecolor='black', linewidth=2,
                  label='Best sparsity at acceptable accuracy', zorder=10)

        ax.set_xlabel('Sparsity Rate')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_title('Accuracy-Sparsity Frontier')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Epoch', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Training dynamics plot saved to {save_path}")


class MetricBasedScheduler:
    """
    Alternative fully adaptive approach: Use metrics to guide pruning.

    Focuses on maximizing a combined score:
        Score = Sparsity * (1 - AccuracyDrop)

    This naturally balances sparsity vs accuracy.
    """

    def __init__(self, baseline_accuracy: float, warmup_steps: int = 1000):
        self.baseline_accuracy = baseline_accuracy
        self.warmup_steps = warmup_steps

        self.step = 0
        self.lambda_mult = 0.1

        self.best_score = 0.0
        self.best_lambda = 0.1

        self.history = {
            'step': [],
            'accuracy': [],
            'sparsity': [],
            'score': [],
            'lambda': [],
        }

    def step_update(self, step: int, accuracy: float, sparsity_rate: float) -> Dict[str, float]:
        """Update based on combined score."""
        self.step = step

        # Calculate score: higher is better
        acc_retention = accuracy / self.baseline_accuracy
        score = sparsity_rate * acc_retention

        # Record
        self.history['step'].append(step)
        self.history['accuracy'].append(accuracy)
        self.history['sparsity'].append(sparsity_rate)
        self.history['score'].append(score)
        self.history['lambda'].append(self.lambda_mult)

        # Warmup
        if step < self.warmup_steps:
            self.lambda_mult *= 1.05
            return {'multiplier': self.lambda_mult}

        # Adaptation: increase lambda if score improved, decrease if not
        if score > self.best_score:
            self.best_score = score
            self.best_lambda = self.lambda_mult
            # Keep pushing
            self.lambda_mult *= 1.1
        else:
            # Back off toward best known lambda
            self.lambda_mult = 0.9 * self.lambda_mult + 0.1 * self.best_lambda

        # Clip
        self.lambda_mult = np.clip(self.lambda_mult, 1e-4, 20.0)

        return {'multiplier': self.lambda_mult}
