"""
IOI Circuit Discovery with KL Budget Constraint + Momentum-Based Exploration.

Philosophy: EXPLORE, DON'T PANIC! We have unlimited epochs to find optimal lambdas.

Key features:
- Momentum-based AIMD: Smooth exploration with deadband to avoid panic reactions
- Aggressive exploration: Start high (λ=2.0), keep pushing when safe
- Deadband filtering: Ignore tiny KL fluctuations to avoid jitter
- Adaptive LR with momentum: High LR during exploration, gentle decreases
- Patient convergence: 100+ epoch minimum, only stop when truly converged
- Per-component adaptation: Balance sparsity across model parts

Momentum strategy:
- Lambda velocity: Maintains direction of change for smooth exploration
- Consecutive action tracking: Builds momentum when consistently increasing/decreasing
- Gentle recovery: Don't panic on first overshoot, give it time
- Exploration bonus: Extra push when far below budget

Usage:
    # Default: Aggressive exploration with momentum (recommended!)
    python ioi_llama_kl_budget.py --kl-budget 0.5

    # With Weights & Biases logging (comprehensive tracking + visualization)
    python ioi_llama_kl_budget.py --kl-budget 0.5 --wandb --wandb-project my-circuits

    # With task loss for better task accuracy
    python ioi_llama_kl_budget.py --kl-budget 0.5 --use-task-loss --task-lambda 1.0

    # Ultra-aggressive exploration (go wild!)
    python ioi_llama_kl_budget.py --kl-budget 0.5 --initial-lambda 5.0 --lambda-add-inc 0.15

    # More conservative (gentler exploration)
    python ioi_llama_kl_budget.py --kl-budget 0.5 --initial-lambda 1.0 --lambda-add-inc 0.04

    # Full setup with W&B tracking
    python ioi_llama_kl_budget.py \
        --kl-budget 0.5 \
        --use-task-loss \
        --flash-attn \
        --wandb \
        --wandb-project circuit-discovery \
        --wandb-name llama-ioi-kl05 \
        --wandb-tags llama ioi momentum

    # Dry run for testing
    python ioi_llama_kl_budget.py --kl-budget 0.5 --dry-run
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
# KL BUDGET SCHEDULER
# ==============================================================================

@dataclass
class KLBudgetConfig:
    """Configuration for KL budget scheduler."""
    warmup_steps: int = 10

    # KL budget constraint
    kl_budget: float = 0.5  # Maximum allowed KL divergence
    kl_tolerance: float = 0.05  # Tolerance around budget

    # Task loss (optional)
    use_task_loss: bool = False
    task_lambda: float = 1.0

    # Adaptation parameters - with momentum
    lambda_additive_increase: float = 0.08  # More aggressive exploration
    lambda_multiplicative_decrease: float = 0.85  # Less panic on decrease
    window_size: int = 15  # Larger window for stability

    # Momentum for smooth adaptation
    use_momentum: bool = True
    momentum_beta: float = 0.8  # High momentum for smooth changes
    lambda_momentum_strength: float = 0.3  # How much momentum affects lambda

    # Exploration bonus
    exploration_bonus: float = 0.02  # Extra push when far from budget
    kl_deadband: float = 0.02  # Don't react to tiny KL changes

    # Initial lambda
    initial_lambda: float = 2.0  # Start high for aggressive exploration

    # Adaptive learning rate with momentum
    use_adaptive_lr: bool = True
    base_lr: float = 1e-3
    lr_increase_factor: float = 1.15  # Gentler increases
    lr_decrease_factor: float = 0.85  # Gentler decreases
    min_lr: float = 5e-4
    max_lr: float = 1e-2  # Allow higher LR for exploration

    # Per-component adaptive sparsity
    use_component_adaptation: bool = True
    component_adaptation_rate: float = 0.15

    # Block-level pruning control (less chaotic)
    block_pruning_start_epoch: int = 2000  # Start block pruning after this epoch
    block_adaptation_frequency: int = 20  # Adapt block lambdas every N epochs (vs every epoch for other components)

    # Convergence - no rush, find optimal
    convergence_patience: int = 100  # Much more patient
    min_training_epochs: int = 100  # Allow more exploration
    max_training_epochs: int = 1000  # Cap at 1000

    # Lambda bounds - wider range
    min_lambda: float = 1e-3
    max_lambda: float = 50.0  # Allow very high sparsity pressure

    # EMA smoothing
    ema_alpha: float = 0.8  # Higher for smoother, less panicky behavior


class KLBudgetScheduler:
    """
    Scheduler that maximizes sparsity while keeping KL divergence within budget.

    Strategy:
    - If KL < budget: Increase sparsity (push for more pruning)
    - If KL > budget: Decrease sparsity (recover KL)
    - If KL ≈ budget: Fine-tune (explore/maintain)
    """

    def __init__(self, config: KLBudgetConfig):
        self.config = config
        self.kl_budget = config.kl_budget

        print(f"\n🎯 Mode: KL BUDGET CONSTRAINT (Adaptive AIMD Strategy)")
        print(f"   KL Budget: {self.kl_budget:.4f}")
        print(f"   Tolerance: ±{config.kl_tolerance:.4f}")
        print(f"   Acceptable range: [{self.kl_budget - config.kl_tolerance:.4f}, {self.kl_budget + config.kl_tolerance:.4f}]")
        print(f"   Task Loss: {'Enabled (λ=' + str(config.task_lambda) + ')' if config.use_task_loss else 'Disabled'}")
        print(f"   AIMD: Additive increase={config.lambda_additive_increase:.4f}, Multiplicative decrease={config.lambda_multiplicative_decrease:.2f}")
        print(f"   Adaptive LR: {'Enabled' if config.use_adaptive_lr else 'Disabled'}")
        print(f"   Component Adaptation: {'Enabled' if config.use_component_adaptation else 'Disabled'}")
        print(f"   Will maximize sparsity within this KL budget")

        # State
        self.step = 0
        self.epoch = 0

        # EMA metrics
        self.ema_accuracy = 0.0
        self.ema_sparsity = 0.0
        self.ema_kl_loss = 0.0

        # Windows for trend detection
        self.kl_window = deque(maxlen=config.window_size)
        self.sparsity_window = deque(maxlen=config.window_size)

        # Lambda state with momentum
        self.lambda_multiplier = config.initial_lambda  # Start higher for faster convergence
        self.best_lambda = config.initial_lambda
        self.lambda_velocity = 0.0  # Momentum for lambda changes
        self.lambda_history = deque(maxlen=20)  # Track lambda trajectory

        # Per-component lambda multipliers (initialized later)
        self.component_lambdas = {}

        # Learning rate state with momentum
        self.current_lr = config.base_lr
        self.lr_velocity = 0.0  # Momentum for LR changes
        self.lr_window = deque(maxlen=5)  # Track LR stability

        # Best tracking (highest sparsity within budget)
        self.best_sparsity = 0.0
        self.best_accuracy = 0.0
        self.best_kl = float('inf')
        self.epochs_since_improvement = 0

        # Exploration tracking
        self.consecutive_increases = 0
        self.consecutive_decreases = 0

        # Phase tracking
        self.phase = "warmup"

        # History - comprehensive tracking
        self.history = {
            'step': [], 'epoch': [], 'accuracy': [], 'sparsity': [],
            'kl_loss': [], 'lambda_mult': [], 'phase': [], 'action': [],
            'lr': [], 'lambda_velocity': [], 'lr_velocity': [],
            'consecutive_increases': [], 'consecutive_decreases': [],
            'kl_gap': [], 'kl_trend': [], 'sparsity_trend': [],
        }

        # Component-level history
        self.component_history = {}

        # Layer-level history
        self.layer_history = {}

    def step_update(
        self,
        step: int,
        epoch: int,
        accuracy: float,
        sparsity_rate: float,
        kl_loss: float,
        component_sparsity: Dict[str, float] = None,
        model=None,
    ) -> Dict[str, float]:
        """Update and return lambda multiplier."""
        self.step = step
        self.epoch = epoch

        # Update EMAs
        alpha = self.config.ema_alpha
        if self.ema_accuracy == 0.0:  # First update
            self.ema_accuracy = accuracy
            self.ema_sparsity = sparsity_rate
            self.ema_kl_loss = kl_loss
        else:
            self.ema_accuracy = alpha * self.ema_accuracy + (1 - alpha) * accuracy
            self.ema_sparsity = alpha * self.ema_sparsity + (1 - alpha) * sparsity_rate
            self.ema_kl_loss = alpha * self.ema_kl_loss + (1 - alpha) * kl_loss

        # Update windows
        self.kl_window.append(kl_loss)
        self.sparsity_window.append(sparsity_rate)

        # Determine action
        action = self._determine_action()

        # Adjust lambda
        old_lambda = self.lambda_multiplier
        self.lambda_multiplier = self._adjust_lambda(action)

        # Adapt per-component lambdas if enabled
        if component_sparsity and model and self.config.use_component_adaptation:
            self.adapt_component_lambdas(model, component_sparsity)

        # Adapt learning rate
        if self.config.use_adaptive_lr:
            self._adapt_learning_rate(action)

        # Calculate trends for logging
        kl_trend = self._calculate_trend(self.kl_window) if len(self.kl_window) >= 3 else 0.0
        sparsity_trend = self._calculate_trend(self.sparsity_window) if len(self.sparsity_window) >= 3 else 0.0
        kl_gap = self.ema_kl_loss - self.kl_budget

        # Record comprehensive metrics
        self.history['step'].append(step)
        self.history['epoch'].append(epoch)
        self.history['accuracy'].append(accuracy)
        self.history['sparsity'].append(sparsity_rate)
        self.history['kl_loss'].append(kl_loss)
        self.history['lambda_mult'].append(self.lambda_multiplier)
        self.history['phase'].append(self.phase)
        self.history['action'].append(action)
        self.history['lr'].append(self.current_lr)
        self.history['lambda_velocity'].append(self.lambda_velocity)
        self.history['lr_velocity'].append(self.lr_velocity)
        self.history['consecutive_increases'].append(self.consecutive_increases)
        self.history['consecutive_decreases'].append(self.consecutive_decreases)
        self.history['kl_gap'].append(kl_gap)
        self.history['kl_trend'].append(kl_trend)
        self.history['sparsity_trend'].append(sparsity_trend)

        # Logging - more frequent early on
        log_freq = 5 if epoch < 50 else 10
        if epoch % log_freq == 0:
            self._log_status(action, old_lambda)

        return {
            'multiplier': self.lambda_multiplier,
            'lr': self.current_lr,
            'component_lambdas': self.component_lambdas.copy() if self.component_lambdas else {}
        }

    def _determine_action(self) -> str:
        """
        Determine action based on KL budget with deadband and momentum awareness.

        Strategy: Explore aggressively, use deadband to avoid panic, maintain momentum.
        """
        # Warmup - very short, just to initialize
        if self.step < self.config.warmup_steps:
            self.phase = "warmup"
            return "warmup"

        # Start adapting immediately after warmup
        if len(self.kl_window) < 2:
            return "increase"  # Start pushing sparsity immediately

        # Calculate trends
        kl_trend = self._calculate_trend(self.kl_window)
        sparsity_trend = self._calculate_trend(self.sparsity_window)

        # KL relative to budget
        kl_gap = self.ema_kl_loss - self.kl_budget
        kl_gap_abs = abs(kl_gap)

        # Deadband: ignore tiny fluctuations
        if kl_gap_abs < self.config.kl_deadband:
            self.phase = "at_budget"
            # Keep momentum going if we have it
            if self.lambda_velocity > 0.01:
                return "explore_momentum"  # Continue increasing with momentum
            else:
                return "maintain"

        # Update best sparsity (relaxed criteria)
        within_budget = self.ema_kl_loss <= self.kl_budget + self.config.kl_tolerance
        if within_budget:
            if self.ema_sparsity > self.best_sparsity + 0.005:  # 0.5% improvement
                self.best_sparsity = self.ema_sparsity
                self.best_accuracy = self.ema_accuracy
                self.best_kl = self.ema_kl_loss
                self.best_lambda = self.lambda_multiplier
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
        else:
            self.epochs_since_improvement += 1

        # Decision logic with momentum consideration
        if kl_gap > 2 * self.config.kl_tolerance:
            # Significantly above budget - but check momentum first
            self.consecutive_decreases += 1
            self.consecutive_increases = 0
            self.phase = "recovery"

            # If we've been decreasing a lot, maybe ease up
            if self.consecutive_decreases > 5:
                return "decrease_gentle"
            else:
                return "decrease"

        elif kl_gap > self.config.kl_tolerance:
            # Above budget - careful
            self.consecutive_decreases += 1
            self.consecutive_increases = 0
            self.phase = "fine_tuning"

            # Check if we're making progress
            if kl_trend < -0.01:  # KL is decreasing, we're recovering
                return "maintain"  # Don't overcorrect
            else:
                return "decrease_gentle"

        elif kl_gap < -2 * self.config.kl_tolerance:
            # Well below budget - explore aggressively!
            self.consecutive_increases += 1
            self.consecutive_decreases = 0
            self.phase = "exploration"

            # Add exploration bonus if we've been increasing successfully
            if self.consecutive_increases > 3:
                return "increase_aggressive"  # Keep the momentum!
            else:
                return "increase"

        elif kl_gap < -self.config.kl_tolerance:
            # Below budget - increase pruning
            self.consecutive_increases += 1
            self.consecutive_decreases = 0
            self.phase = "exploitation"
            return "increase"

        else:
            # At budget (within tolerance) - explore carefully
            self.phase = "fine_tuning"

            # Random exploration to find better lambda
            if np.random.random() < 0.3:  # 30% chance
                return "explore"
            else:
                # Maintain momentum if we have it
                if abs(self.lambda_velocity) > 0.01:
                    return "maintain_momentum"
                else:
                    return "maintain"

    def _adjust_lambda(self, action: str) -> float:
        """
        Adjust lambda with momentum for smooth, exploratory adaptation.

        Strategy: AIMD with momentum - keep exploring, don't panic!
        """
        add_inc = self.config.lambda_additive_increase
        mult_dec = self.config.lambda_multiplicative_decrease
        exploration_bonus = self.config.exploration_bonus

        # Compute base change
        if action == 'warmup':
            delta = add_inc
        elif action == 'increase':
            delta = add_inc
        elif action == 'increase_aggressive':
            delta = add_inc * 1.5 + exploration_bonus  # Extra aggressive!
        elif action == 'explore':
            delta = add_inc * 0.5
        elif action == 'explore_momentum':
            delta = add_inc * 0.3  # Small exploration
        elif action == 'decrease':
            delta = self.lambda_multiplier * (mult_dec - 1)  # Negative delta
        elif action == 'decrease_gentle':
            delta = self.lambda_multiplier * (0.95 - 1)  # Gentler decrease
        elif action == 'maintain':
            delta = 0.0
        elif action == 'maintain_momentum':
            delta = self.lambda_velocity * 0.5  # Use existing momentum
        else:
            delta = 0.0

        # Apply momentum if enabled
        if self.config.use_momentum:
            # Update velocity with momentum
            beta = self.config.momentum_beta
            self.lambda_velocity = beta * self.lambda_velocity + (1 - beta) * delta

            # Apply velocity to lambda
            momentum_contribution = self.lambda_velocity * self.config.lambda_momentum_strength
            new_lambda = self.lambda_multiplier + delta + momentum_contribution
        else:
            new_lambda = self.lambda_multiplier + delta

        # Track lambda history
        self.lambda_history.append(new_lambda)

        # Clip to bounds
        new_lambda = np.clip(new_lambda, self.config.min_lambda, self.config.max_lambda)

        return new_lambda

    def _adapt_learning_rate(self, action: str):
        """
        Adapt learning rate with momentum - smooth exploration, not panic!

        Strategy: Increase when exploring, keep high when stable, decrease gently when needed
        """
        # Track KL variance for stability
        if len(self.kl_window) >= 5:
            kl_variance = np.var(list(self.kl_window))
            is_stable = kl_variance < 0.02  # More relaxed stability threshold
        else:
            is_stable = True  # Assume stable initially

        # Compute LR change based on phase
        lr_delta = 0.0

        if self.phase == "exploration":
            # Exploring - keep LR high
            if is_stable:
                lr_delta = self.current_lr * (self.config.lr_increase_factor - 1)
            else:
                lr_delta = 0  # Maintain current LR even if unstable during exploration

        elif self.phase == "at_budget":
            # At budget - maintain high LR for continued exploration
            lr_delta = 0  # Keep exploring at current pace

        elif self.phase == "recovery":
            # Recovery - decrease LR only if repeatedly overshooting
            if self.consecutive_decreases > 3:
                lr_delta = self.current_lr * (self.config.lr_decrease_factor - 1)
            else:
                lr_delta = 0  # Don't panic on first overshoot

        elif self.phase == "fine_tuning":
            # Fine-tuning - very gentle decrease
            lr_delta = self.current_lr * (0.99 - 1)  # 1% decrease

        # Apply momentum to LR
        if self.config.use_momentum:
            beta = self.config.momentum_beta
            self.lr_velocity = beta * self.lr_velocity + (1 - beta) * lr_delta
            self.current_lr = self.current_lr + self.lr_velocity
        else:
            self.current_lr = self.current_lr + lr_delta

        # Clip to bounds
        self.current_lr = np.clip(self.current_lr, self.config.min_lr, self.config.max_lr)

    def adapt_component_lambdas(self, model, current_sparsity_by_component: Dict[str, float]):
        """
        Adapt per-component lambda multipliers independently using AIMD.

        Strategy: Each component tries to maximize its own sparsity.
        - If component is still dense (low sparsity): increase lambda to prune more
        - If component stopped improving: back off lambda slightly
        - No coordination between components - find optimal lambda independently
        """
        if not self.config.use_component_adaptation:
            return {}

        # Initialize component lambdas and tracking if needed
        if not self.component_lambdas:
            for comp_name in current_sparsity_by_component.keys():
                self.component_lambdas[comp_name] = 1.0

        # Track previous sparsity for each component
        if not hasattr(self, 'prev_component_sparsity'):
            self.prev_component_sparsity = {}

        # Adapt each component independently
        for comp_name, comp_sparsity in current_sparsity_by_component.items():
            prev_sparsity = self.prev_component_sparsity.get(comp_name, 0.0)
            sparsity_increase = comp_sparsity - prev_sparsity

            current_lambda = self.component_lambdas[comp_name]

            # AIMD per component:
            # - If sparsity is increasing: keep increasing lambda (additive)
            # - If sparsity plateaued or decreased: decrease lambda (multiplicative)
            # - If very dense (< 0.3): aggressive increase
            # - If very sparse (> 0.9): gentle decrease to avoid collapse

            if comp_sparsity < 0.3:
                # Very dense - aggressive increase
                self.component_lambdas[comp_name] = current_lambda + 0.3
            elif sparsity_increase > 0.01:
                # Making progress - additive increase
                self.component_lambdas[comp_name] = current_lambda + self.config.component_adaptation_rate
            elif sparsity_increase < -0.01:
                # Regressing - multiplicative decrease
                self.component_lambdas[comp_name] = current_lambda * (1 - self.config.component_adaptation_rate)
            elif comp_sparsity > 0.9:
                # Very sparse - gentle decrease
                self.component_lambdas[comp_name] = current_lambda * 0.95
            # else: plateaued - maintain current lambda

            # Clip to reasonable bounds (wider range than before)
            self.component_lambdas[comp_name] = np.clip(
                self.component_lambdas[comp_name], 0.05, 10.0
            )

            # Update previous sparsity
            self.prev_component_sparsity[comp_name] = comp_sparsity

        # Apply component lambdas to model's pruning config
        # Map component names to config lambda attributes
        component_to_config = {
            'attention_heads': 'lambda_attention_heads',
            'attention_neurons': 'lambda_attention_neurons',
            'mlp_hidden': 'lambda_mlp_hidden',
            'mlp_output': 'lambda_mlp_output',
            'attention_blocks': 'lambda_attention_blocks',
            'mlp_blocks': 'lambda_mlp_blocks',
            'full_layers': 'lambda_full_layers',
        }

        for comp_name, comp_lambda in self.component_lambdas.items():
            if comp_name in component_to_config:
                config_attr = component_to_config[comp_name]
                # Apply both global lambda multiplier AND component-specific lambda
                setattr(model.pruning_config, config_attr, comp_lambda * self.lambda_multiplier)

        # Track component history
        for comp_name, comp_sparsity in current_sparsity_by_component.items():
            if comp_name not in self.component_history:
                self.component_history[comp_name] = {'sparsity': [], 'lambda': []}
            self.component_history[comp_name]['sparsity'].append(comp_sparsity)
            self.component_history[comp_name]['lambda'].append(self.component_lambdas.get(comp_name, 1.0))

        return self.component_lambdas

    def log_to_wandb(self, epoch: int, component_sparsity: Dict[str, float] = None,
                     layer_sparsity: Dict[str, float] = None):
        """Log metrics to Weights & Biases if available."""
        try:
            import wandb
            if not wandb.run:
                return

            # Main metrics
            metrics = {
                'epoch': epoch,
                'scheduler/lambda': self.lambda_multiplier,
                'scheduler/lr': self.current_lr,
                'scheduler/lambda_velocity': self.lambda_velocity,
                'scheduler/lr_velocity': self.lr_velocity,
                'metrics/kl_loss': self.ema_kl_loss,
                'metrics/accuracy': self.ema_accuracy,
                'metrics/sparsity': self.ema_sparsity,
                'metrics/kl_gap': self.ema_kl_loss - self.kl_budget,
                'scheduler/consecutive_increases': self.consecutive_increases,
                'scheduler/consecutive_decreases': self.consecutive_decreases,
                'best/sparsity': self.best_sparsity,
                'best/accuracy': self.best_accuracy,
                'best/kl': self.best_kl,
                'best/lambda': self.best_lambda,
            }

            # Trends if available
            if len(self.kl_window) >= 3:
                metrics['trends/kl'] = self._calculate_trend(self.kl_window)
                metrics['trends/sparsity'] = self._calculate_trend(self.sparsity_window)

            # Component-level metrics
            if component_sparsity:
                for comp_name, comp_sparse in component_sparsity.items():
                    metrics[f'components/{comp_name}/sparsity'] = comp_sparse
                    if comp_name in self.component_lambdas:
                        metrics[f'components/{comp_name}/lambda'] = self.component_lambdas[comp_name]

            # Layer-level metrics
            if layer_sparsity:
                for layer_name, layer_sparse in layer_sparsity.items():
                    metrics[f'layers/{layer_name}/sparsity'] = layer_sparse

                # Track layer history
                for layer_name, layer_sparse in layer_sparsity.items():
                    if layer_name not in self.layer_history:
                        self.layer_history[layer_name] = []
                    self.layer_history[layer_name].append(layer_sparse)

            # Phase as category
            metrics['scheduler/phase'] = self.phase

            wandb.log(metrics, step=epoch)

        except ImportError:
            pass  # wandb not installed

    def _calculate_trend(self, window: deque) -> float:
        """Calculate trend (slope) of recent values."""
        if len(window) < 3:
            return 0.0
        values = list(window)
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def should_stop_early(self) -> bool:
        """
        Check convergence - but be patient! We want to find optimal lambdas.

        Only stop if we've truly converged or hit max epochs.
        """
        # Never stop before minimum epochs
        if self.epoch < self.config.min_training_epochs:
            return False

        # Stop at max epochs (hard limit)
        if self.epoch >= self.config.max_training_epochs:
            print(f"\n⏰ Reached max epochs ({self.config.max_training_epochs})")
            return True

        # Only stop if no improvement for a LONG time
        if self.epochs_since_improvement > self.config.convergence_patience:
            print(f"\n📊 No improvement for {self.config.convergence_patience} epochs")
            return True

        # Very strict convergence criteria - lambda and metrics must be stable
        if len(self.lambda_history) >= 20:
            lambda_std = np.std(list(self.lambda_history)[-20:])
            sparsity_std = np.std(list(self.sparsity_window))
            kl_std = np.std(list(self.kl_window))

            # All metrics stable AND at budget
            all_stable = lambda_std < 0.05 and sparsity_std < 0.005 and kl_std < 0.005
            at_budget = abs(self.ema_kl_loss - self.kl_budget) < self.config.kl_tolerance

            if all_stable and at_budget:
                print(f"\n✅ Converged: λ_std={lambda_std:.4f}, sparsity_std={sparsity_std:.4f}, kl_std={kl_std:.4f}")
                return True

        return False

    def get_final_summary(self) -> Dict:
        """Get final summary."""
        return {
            'kl_budget': self.kl_budget,
            'final_accuracy': self.ema_accuracy,
            'final_sparsity': self.ema_sparsity,
            'final_kl': self.ema_kl_loss,
            'best_sparsity': self.best_sparsity,
            'best_accuracy': self.best_accuracy,
            'best_kl': self.best_kl,
            'best_lambda': self.best_lambda,
            'total_epochs': self.epoch,
            'phase': self.phase,
            'kl_within_budget': self.ema_kl_loss <= self.kl_budget + self.config.kl_tolerance,
        }

    def _log_status(self, action: str, old_lambda: float):
        """Print status."""
        kl_gap = self.ema_kl_loss - self.kl_budget

        print(f"\n[Scheduler] Epoch {self.epoch} | Phase: {self.phase}")
        print(f"  KL Loss:  {self.ema_kl_loss:.4f} (budget: {self.kl_budget:.4f}, gap: {kl_gap:+.4f})")
        print(f"  Accuracy: {self.ema_accuracy:.4f}")
        print(f"  Sparsity: {self.ema_sparsity:.4f} (best: {self.best_sparsity:.4f})")
        print(f"  Lambda:   {old_lambda:.3f} → {self.lambda_multiplier:.3f} (action: {action})")
        if self.config.use_adaptive_lr:
            print(f"  LR:       {self.current_lr:.2e}")

        if self.ema_kl_loss <= self.kl_budget + self.config.kl_tolerance:
            print(f"  ✅ Within budget")
        else:
            print(f"  ⚠️  Exceeds budget")

    def plot_training_dynamics(self, save_dir: str):
        """Generate comprehensive plots of training dynamics."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("⚠️  Matplotlib not installed, skipping plots")
            return

        os.makedirs(save_dir, exist_ok=True)
        epochs = np.array(self.history['epoch'])

        # ============ MAIN DASHBOARD (4x3 grid) ============
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: KL, Sparsity, Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history['kl_loss'], alpha=0.7, linewidth=2, label='KL Loss')
        ax1.axhline(self.kl_budget, color='blue', linestyle='--', linewidth=2, label='Budget')
        ax1.axhline(self.kl_budget + self.config.kl_tolerance, color='red', linestyle=':', alpha=0.6)
        ax1.axhline(self.kl_budget - self.config.kl_tolerance, color='green', linestyle=':', alpha=0.6)
        ax1.fill_between(epochs, self.kl_budget - self.config.kl_tolerance,
                         self.kl_budget + self.config.kl_tolerance, alpha=0.1, color='blue')
        if self.best_kl != float('inf'):
            ax1.scatter([epochs[np.argmax(self.history['sparsity'])]], [self.best_kl],
                       color='red', s=200, marker='*', edgecolor='black', linewidth=2, zorder=10)
        ax1.set_ylabel('KL Divergence', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('KL Divergence vs Budget', fontsize=14, fontweight='bold')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history['sparsity'], alpha=0.7, linewidth=2, color='orange', label='Sparsity')
        if self.best_sparsity > 0:
            best_epoch = np.argmax(self.history['sparsity'])
            ax2.scatter([epochs[best_epoch]], [self.best_sparsity],
                       color='red', s=200, marker='*', edgecolor='black', linewidth=2, zorder=10)
            ax2.axhline(self.best_sparsity, color='green', linestyle='--', alpha=0.6, label=f'Best: {self.best_sparsity:.3f}')
        ax2.set_ylabel('Sparsity', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Sparsity Over Time', fontsize=14, fontweight='bold')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, self.history['accuracy'], alpha=0.7, linewidth=2, color='green', label='Accuracy')
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')

        # Row 2: Lambda, LR, Phase Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(epochs, self.history['lambda_mult'], alpha=0.7, linewidth=2, color='purple', label='Lambda')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(epochs, self.history['lambda_velocity'], alpha=0.5, linewidth=1,
                     color='red', linestyle='--', label='Velocity')
        ax4.set_ylabel('Lambda Multiplier', fontsize=12, color='purple')
        ax4_twin.set_ylabel('Lambda Velocity', fontsize=12, color='red')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left', fontsize=10)
        ax4_twin.legend(loc='upper right', fontsize=10)
        ax4.set_title('Lambda with Momentum', fontsize=14, fontweight='bold')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(epochs, self.history['lr'], alpha=0.7, linewidth=2, color='teal', label='Learning Rate')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(epochs, self.history['lr_velocity'], alpha=0.5, linewidth=1,
                     color='orange', linestyle='--', label='LR Velocity')
        ax5.set_ylabel('Learning Rate', fontsize=12, color='teal')
        ax5_twin.set_ylabel('LR Velocity', fontsize=12, color='orange')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper left', fontsize=10)
        ax5_twin.legend(loc='upper right', fontsize=10)
        ax5.set_title('Learning Rate Adaptation', fontsize=14, fontweight='bold')

        ax6 = fig.add_subplot(gs[1, 2])
        phases = self.history['phase']
        unique_phases = list(set(phases))
        phase_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_phases)))
        phase_to_color = {phase: color for phase, color in zip(unique_phases, phase_colors)}
        for i in range(len(epochs)):
            ax6.axvspan(epochs[i]-0.5 if i > 0 else 0, epochs[i]+0.5,
                       color=phase_to_color[phases[i]], alpha=0.6)
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('Phase', fontsize=12)
        ax6.set_yticks(range(len(unique_phases)))
        ax6.set_yticklabels(unique_phases, fontsize=9)
        ax6.set_title('Training Phase Evolution', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # Row 3: Momentum metrics, KL gap, trends
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(epochs, self.history['consecutive_increases'], alpha=0.7, linewidth=2,
                color='green', label='Consecutive Increases')
        ax7.plot(epochs, self.history['consecutive_decreases'], alpha=0.7, linewidth=2,
                color='red', label='Consecutive Decreases')
        ax7.set_ylabel('Count', fontsize=12)
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax7.set_title('Momentum Tracking', fontsize=14, fontweight='bold')

        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(epochs, self.history['kl_gap'], alpha=0.7, linewidth=2, color='brown')
        ax8.axhline(0, color='blue', linestyle='--', linewidth=2, label='Budget')
        ax8.axhline(self.config.kl_tolerance, color='red', linestyle=':', alpha=0.6)
        ax8.axhline(-self.config.kl_tolerance, color='green', linestyle=':', alpha=0.6)
        ax8.fill_between(epochs, -self.config.kl_tolerance, self.config.kl_tolerance,
                        alpha=0.1, color='blue')
        ax8.set_ylabel('KL Gap (KL - Budget)', fontsize=12)
        ax8.set_xlabel('Epoch', fontsize=12)
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.set_title('KL Gap from Budget', fontsize=14, fontweight='bold')

        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(epochs, self.history['kl_trend'], alpha=0.7, linewidth=2, color='blue', label='KL Trend')
        ax9.plot(epochs, self.history['sparsity_trend'], alpha=0.7, linewidth=2, color='orange', label='Sparsity Trend')
        ax9.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax9.set_ylabel('Trend (Slope)', fontsize=12)
        ax9.set_xlabel('Epoch', fontsize=12)
        ax9.legend(fontsize=10)
        ax9.grid(True, alpha=0.3)
        ax9.set_title('Metric Trends', fontsize=14, fontweight='bold')

        # Row 4: Sparsity vs KL scatter, Action distribution
        ax10 = fig.add_subplot(gs[3, :2])
        scatter = ax10.scatter(self.history['kl_loss'], self.history['sparsity'],
                              c=epochs, cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax10.axvline(self.kl_budget, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Budget')
        ax10.axvline(self.kl_budget + self.config.kl_tolerance, color='red', linestyle=':', alpha=0.5)
        ax10.axvline(self.kl_budget - self.config.kl_tolerance, color='green', linestyle=':', alpha=0.5)
        if self.best_sparsity > 0:
            ax10.scatter([self.best_kl], [self.best_sparsity], color='red', s=400, marker='*',
                        edgecolor='black', linewidth=3, zorder=10, label='Best Point')
        ax10.set_xlabel('KL Divergence', fontsize=12)
        ax10.set_ylabel('Sparsity', fontsize=12)
        ax10.legend(fontsize=10)
        ax10.grid(True, alpha=0.3)
        ax10.set_title('Sparsity vs KL Tradeoff Curve', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax10)
        cbar.set_label('Epoch', fontsize=12)

        ax11 = fig.add_subplot(gs[3, 2])
        actions = self.history['action']
        unique_actions = list(set(actions))
        action_counts = {action: actions.count(action) for action in unique_actions}
        ax11.bar(range(len(unique_actions)), list(action_counts.values()),
                color=plt.cm.tab20(np.linspace(0, 1, len(unique_actions))))
        ax11.set_xticks(range(len(unique_actions)))
        ax11.set_xticklabels(unique_actions, rotation=45, ha='right', fontsize=9)
        ax11.set_ylabel('Count', fontsize=12)
        ax11.set_title('Action Distribution', fontsize=14, fontweight='bold')
        ax11.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Training Dynamics Dashboard', fontsize=18, fontweight='bold', y=0.995)
        fig.savefig(f'{save_dir}/training_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Dashboard saved: {save_dir}/training_dashboard.png")

        # ============ COMPONENT-LEVEL PLOTS ============
        if self.component_history:
            self._plot_component_dynamics(save_dir)

    def _plot_component_dynamics(self, save_dir: str):
        """Plot component-level sparsity and lambda evolution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        n_components = len(self.component_history)
        if n_components == 0:
            return

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Component sparsity over time
        ax = axes[0]
        for comp_name, comp_data in self.component_history.items():
            ax.plot(comp_data['sparsity'], label=comp_name, linewidth=2, alpha=0.7)
        ax.set_ylabel('Sparsity', fontsize=12)
        ax.set_xlabel('Evaluation Step', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Component-Level Sparsity Evolution', fontsize=14, fontweight='bold')

        # Component lambda over time
        ax = axes[1]
        for comp_name, comp_data in self.component_history.items():
            ax.plot(comp_data['lambda'], label=comp_name, linewidth=2, alpha=0.7)
        ax.set_ylabel('Lambda Multiplier', fontsize=12)
        ax.set_xlabel('Evaluation Step', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Component-Level Lambda Adaptation', fontsize=14, fontweight='bold')

        plt.tight_layout()
        fig.savefig(f'{save_dir}/component_dynamics.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Component dynamics saved: {save_dir}/component_dynamics.png")

        # ============ LAYER-WISE SPARSITY HEATMAP ============
        if self.layer_history:
            self._plot_layer_heatmap(save_dir)

    def _plot_layer_heatmap(self, save_dir: str):
        """Plot layer-wise sparsity as a heatmap over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not self.layer_history:
            return

        # Get layer names sorted by layer number
        layer_names = sorted(self.layer_history.keys(), key=lambda x: int(x.split('_')[1]))
        n_layers = len(layer_names)
        n_steps = len(self.layer_history[layer_names[0]])

        if n_layers == 0 or n_steps == 0:
            return

        # Build matrix: rows = layers, cols = timesteps
        sparsity_matrix = np.zeros((n_layers, n_steps))
        for i, layer_name in enumerate(layer_names):
            sparsity_matrix[i, :] = self.layer_history[layer_name]

        fig, ax = plt.subplots(1, 1, figsize=(16, max(8, n_layers * 0.3)))

        # Heatmap
        im = ax.imshow(sparsity_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                      interpolation='nearest')

        # Labels
        ax.set_xlabel('Evaluation Step', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f'L{i}' for i in range(n_layers)], fontsize=9)
        ax.set_title('Layer-Wise Sparsity Evolution Over Time', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sparsity', fontsize=12)

        # Grid
        ax.set_xticks(np.arange(0, n_steps, max(1, n_steps // 10)))
        ax.grid(False)

        plt.tight_layout()
        fig.savefig(f'{save_dir}/layer_sparsity_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Layer heatmap saved: {save_dir}/layer_sparsity_heatmap.png")


# ==============================================================================
# Helper Functions
# ==============================================================================

@dataclass
class KLBudgetLlamaPruningConfig(PruningConfig):
    """Base config for KL budget training."""
    init_value: float = 2.0
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.0

    prune_attention_heads: bool = True
    lambda_attention_heads: float = 1.0

    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 5.0

    prune_mlp_output: bool = True
    lambda_mlp_output: float = 5.0

    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 1.0

    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.0001

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.0001

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0

    # EMBEDDING GATE COMPLETELY REMOVED - no config needed


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


def compute_component_sparsity(model) -> Dict[str, float]:
    """
    Compute sparsity rate for each component type.

    Component types:
    - attention_heads: Per-head gates in attention
    - attention_neurons: Per-neuron gates in attention (head_dim × num_heads)
    - mlp_hidden: Hidden/intermediate layer neurons (up projection)
    - mlp_output: Output layer neurons (down projection)
    - attention_blocks: Whole attention block gates
    - mlp_blocks: Whole MLP block gates

    Note: Only tracks components that are enabled for pruning in the pruning config.
    """
    from models.l0 import HardConcreteGate

    component_stats = {}

    # Get pruning config to filter out disabled components
    pruning_config = model.pruning_config

    for name, module in model.named_modules():
        if isinstance(module, HardConcreteGate):
            # Categorize by actual gate names from llama_circuit.py
            comp_type = None
            if 'head_gates' in name:
                if pruning_config.prune_attention_heads:
                    comp_type = 'attention_heads'
            elif 'neuron_gates' in name:
                if pruning_config.prune_attention_neurons:
                    comp_type = 'attention_neurons'
            elif 'hidden_gates' in name:
                if pruning_config.prune_mlp_hidden:
                    comp_type = 'mlp_hidden'
            elif 'output_gates' in name:
                if pruning_config.prune_mlp_output:
                    comp_type = 'mlp_output'
            elif 'attention_block_gate' in name:
                if pruning_config.prune_attention_blocks:
                    comp_type = 'attention_blocks'
            elif 'mlp_block_gate' in name:
                if pruning_config.prune_mlp_blocks:
                    comp_type = 'mlp_blocks'
            # NO EMBEDDING GATE - completely removed from model
            elif 'layer_gates' in name:
                if pruning_config.prune_full_layers:
                    comp_type = 'full_layers'

            # Skip if this component type is not being pruned
            if comp_type is None:
                continue

            if comp_type not in component_stats:
                component_stats[comp_type] = {'total': 0, 'open': 0}

            with torch.no_grad():
                gates = module()
                component_stats[comp_type]['total'] += gates.numel()
                component_stats[comp_type]['open'] += (gates > 0.5).sum().item()

    # Compute sparsity for each component
    component_sparsity = {}
    for comp_type, stats in component_stats.items():
        if stats['total'] > 0:
            component_sparsity[comp_type] = 1.0 - (stats['open'] / stats['total'])
        else:
            component_sparsity[comp_type] = 0.0

    return component_sparsity


def compute_layer_sparsity(model) -> Dict[str, float]:
    """
    Compute sparsity for each layer individually.

    Returns dict with keys like 'layer_0', 'layer_1', etc.
    """
    from models.l0 import HardConcreteGate

    layer_stats = {}

    for name, module in model.named_modules():
        if isinstance(module, HardConcreteGate):
            # Extract layer number from name
            # Names look like: model.layers.0.self_attn.head_gates
            parts = name.split('.')

            # Find the layer index
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue

            if layer_idx is not None:
                layer_key = f'layer_{layer_idx}'

                if layer_key not in layer_stats:
                    layer_stats[layer_key] = {'total': 0, 'open': 0}

                with torch.no_grad():
                    gates = module()
                    layer_stats[layer_key]['total'] += gates.numel()
                    layer_stats[layer_key]['open'] += (gates > 0.5).sum().item()

    # Compute sparsity for each layer
    layer_sparsity = {}
    for layer_key, stats in sorted(layer_stats.items(), key=lambda x: int(x[0].split('_')[1])):
        if stats['total'] > 0:
            layer_sparsity[layer_key] = 1.0 - (stats['open'] / stats['total'])
        else:
            layer_sparsity[layer_key] = 0.0

    return layer_sparsity


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KL Budget Circuit Discovery")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints_llama_kl_budget')

    # Key parameter: KL budget
    parser.add_argument('--kl-budget', type=float, required=True,
                        help='Maximum allowed KL divergence (e.g., 0.5)')
    parser.add_argument('--kl-tolerance', type=float, default=0.05,
                        help='Tolerance around KL budget')

    # Task loss (optional)
    parser.add_argument('--use-task-loss', action='store_true',
                        help='Include task loss for additional correctness constraint')
    parser.add_argument('--task-lambda', type=float, default=1.0,
                        help='Weight for task loss (default: 1.0)')

    # Lambda adaptation (AIMD)
    parser.add_argument('--lambda-add-inc', type=float, default=0.05,
                        help='Additive increase for lambda (default: 0.05)')
    parser.add_argument('--lambda-mult-dec', type=float, default=0.75,
                        help='Multiplicative decrease for lambda (default: 0.75)')
    parser.add_argument('--initial-lambda', type=float, default=1.0,
                        help='Initial lambda value (default: 1.0)')

    # Adaptive features
    parser.add_argument('--no-adaptive-lr', action='store_true',
                        help='Disable adaptive learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-4,
                        help='Minimum learning rate (default: 1e-4)')
    parser.add_argument('--max-lr', type=float, default=1e-2,
                        help='Maximum learning rate (default: 1e-2)')
    parser.add_argument('--no-component-adaptation', action='store_true',
                        help='Disable per-component sparsity adaptation')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping (train for max_epochs)')

    # Speedups
    parser.add_argument('--flash-attn', action='store_true')

    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='circuit-discovery',
                        help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity/team name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=None,
                        help='W&B tags for this run')

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

    # Initialize Weights & Biases
    if args.wandb:
        try:
            import wandb
            wandb_config = {
                'model': args.model,
                'kl_budget': args.kl_budget,
                'kl_tolerance': args.kl_tolerance,
                'initial_lambda': args.initial_lambda,
                'lambda_add_inc': args.lambda_add_inc,
                'lambda_mult_dec': args.lambda_mult_dec,
                'use_task_loss': args.use_task_loss,
                'task_lambda': args.task_lambda if args.use_task_loss else None,
                'base_lr': args.lr,
                'batch_size': args.batch_size,
                'flash_attention': args.flash_attn,
                'adaptive_lr': not args.no_adaptive_lr,
                'component_adaptation': not args.no_component_adaptation,
            }
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                tags=args.wandb_tags,
                config=wandb_config,
            )
            print(f"🔗 W&B run: {wandb.run.url}")
        except ImportError:
            print("⚠️  wandb not installed, skipping W&B logging")
            args.wandb = False

    print("="*80)
    print("  KL BUDGET CIRCUIT DISCOVERY")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Flash Attention: {args.flash_attn}")
    print(f"Task Loss: {'Enabled (λ=' + str(args.task_lambda) + ')' if args.use_task_loss else 'Disabled'}")
    print(f"W&B Logging: {'Enabled' if args.wandb else 'Disabled'}")

    # Load models
    print("\n--- Loading models ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    pruning_config = KLBudgetLlamaPruningConfig()
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

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True)

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
    scheduler_config = KLBudgetConfig(
        warmup_steps=pruning_config.sparsity_warmup_steps,
        kl_budget=args.kl_budget,
        kl_tolerance=args.kl_tolerance,
        use_task_loss=args.use_task_loss,
        task_lambda=args.task_lambda,
        lambda_additive_increase=args.lambda_add_inc,
        lambda_multiplicative_decrease=args.lambda_mult_dec,
        initial_lambda=args.initial_lambda,
        use_adaptive_lr=not args.no_adaptive_lr,
        use_component_adaptation=not args.no_component_adaptation,
        base_lr=args.lr,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
    )
    scheduler = KLBudgetScheduler(scheduler_config)

    # Setup training - adaptive LR will update this
    optimizer = AdamW([p for p in circuit_model.parameters() if p.requires_grad], lr=scheduler.current_lr)

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

    # Track metrics per epoch for adaptive LR
    epoch_kl_losses = []
    epoch_sparsities = []

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        circuit_model.train()

        # Reset epoch metrics
        epoch_kl_losses.clear()
        epoch_sparsities.clear()

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )

            # KL loss - per-token averaged, then per-sample averaged
            total_kl = 0
            num_valid_samples = 0
            for i in range(outputs.logits.size(0)):
                t_start = batch['T_Start'][i].item() - 1
                t_end = batch['T_End'][i].item() - 1

                # Get valid sequence length (before padding)
                valid_length = batch['attention_mask'][i].sum().item()

                # Don't compute KL on padding positions
                end_pos = min(t_end, valid_length)

                if t_start < end_pos:
                    # KL divergence: sum over vocabulary, mean over tokens
                    # This gives per-token KL, independent of sequence length
                    num_tokens = end_pos - t_start
                    kl_per_token = F.kl_div(
                        F.log_softmax(outputs.logits[i, t_start:end_pos, :].float(), dim=-1),
                        F.log_softmax(cached_train_logits[batch_idx][i, t_start:end_pos, :].float(), dim=-1),
                        reduction='batchmean',  # Averages over token dimension
                        log_target=True,
                    )
                    total_kl += kl_per_token
                    num_valid_samples += 1

            # Average over valid samples
            kl_loss = total_kl / num_valid_samples if num_valid_samples > 0 else torch.tensor(0.0, device=outputs.logits.device)

            # Task loss (optional)
            task_loss = 0.0
            if scheduler.config.use_task_loss:
                # Use the same task loss as in ioi_llama_hybrid_adaptive.py
                pos_good = batch['T_Start'] - 1
                token_good = batch['target_tokens'][:, 0]
                pos_bad = batch['D_Start'] - 1
                token_bad = batch['distractor_tokens'][:, 0]
                batch_indices = torch.arange(outputs.logits.size(0), device=DEVICE)

                logit_good = outputs.logits[batch_indices, pos_good, token_good].float()
                logit_bad = outputs.logits[batch_indices, pos_bad, token_bad].float()
                task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean() * scheduler.config.task_lambda

            # Sparsity loss with adaptive multiplier
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']
            sparsity_loss = sparsity_loss * scheduler.lambda_multiplier

            loss = kl_loss + sparsity_loss# + task_loss
            loss.backward()
            optimizer.step()

            # Track metrics for adaptive LR (lightweight, no eval)
            epoch_kl_losses.append(kl_loss.item())

            total_steps += 1

        # Compute epoch-level metrics for adaptive LR (every epoch)
        avg_epoch_kl = np.mean(epoch_kl_losses) if epoch_kl_losses else 0.0

        # Quick sparsity check (lightweight)
        with torch.no_grad():
            epoch_sparsity = compute_overall_sparsity(circuit_model)
            epoch_sparsities.append(epoch_sparsity)

        # Update scheduler metrics and adapt LR every epoch
        scheduler.kl_window.append(avg_epoch_kl)
        scheduler.sparsity_window.append(epoch_sparsity)
        scheduler.epoch = epoch + 1

        # Update EMA for better tracking
        alpha = scheduler.config.ema_alpha
        if scheduler.ema_kl_loss == 0.0:
            scheduler.ema_kl_loss = avg_epoch_kl
            scheduler.ema_sparsity = epoch_sparsity
        else:
            scheduler.ema_kl_loss = alpha * scheduler.ema_kl_loss + (1 - alpha) * avg_epoch_kl
            scheduler.ema_sparsity = alpha * scheduler.ema_sparsity + (1 - alpha) * epoch_sparsity

        # Determine action and adapt LR every epoch
        if scheduler.config.use_adaptive_lr and epoch > 0:
            old_lr = scheduler.current_lr
            action = scheduler._determine_action()
            old_lambda = scheduler.lambda_multiplier
            scheduler.lambda_multiplier = scheduler._adjust_lambda(action)
            scheduler._adapt_learning_rate(action)

            # Apply new LR to optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.current_lr

            # Log LR changes (only when significant)
            if abs(old_lr - scheduler.current_lr) / old_lr > 0.01:  # >1% change
                print(f"Epoch {epoch+1}: LR {old_lr:.2e} → {scheduler.current_lr:.2e} | λ {old_lambda:.2f} → {scheduler.lambda_multiplier:.2f} | {action} | {scheduler.phase} | KL {avg_epoch_kl:.3f}")

        # Log to W&B every epoch (lightweight metrics)
        if args.wandb:
            comp_sparsity = compute_component_sparsity(circuit_model) if scheduler.config.use_component_adaptation else None
            layer_sparsity = compute_layer_sparsity(circuit_model)
            scheduler.log_to_wandb(epoch + 1, comp_sparsity, layer_sparsity)

        # Validation - more frequent early on, less frequent later
        # Early epochs (0-50): every 5 epochs
        # Mid epochs (50-100): every 10 epochs
        # Late epochs (100+): every 20 epochs
        if epoch < 50:
            eval_freq = 5
        elif epoch < 100:
            eval_freq = 10
        else:
            eval_freq = 20

        if (epoch + 1) % eval_freq == 0 or epoch == 0:
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
            comp_sparsity = compute_component_sparsity(circuit_model) if scheduler.config.use_component_adaptation else None

            update_result = scheduler.step_update(
                step=total_steps,
                epoch=epoch + 1,
                accuracy=val_results['accuracy'],
                sparsity_rate=current_sparsity,
                kl_loss=val_results['kl_div'],
                component_sparsity=comp_sparsity,
                model=circuit_model,
            )

            # Update optimizer LR if adaptive LR is enabled
            if scheduler.config.use_adaptive_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = update_result['lr']

            # Check early stopping (unless disabled)
            if not args.no_early_stopping and scheduler.should_stop_early():
                print("\n🎉 Converged! Stopping early.")
                break

    # Save comprehensive plots
    scheduler.plot_training_dynamics(args.save_dir)

    # Upload plots to W&B if enabled
    if args.wandb:
        try:
            import wandb
            dashboard_path = f'{args.save_dir}/training_dashboard.png'
            if os.path.exists(dashboard_path):
                wandb.log({"training_dashboard": wandb.Image(dashboard_path)})
            component_path = f'{args.save_dir}/component_dynamics.png'
            if os.path.exists(component_path):
                wandb.log({"component_dynamics": wandb.Image(component_path)})
            layer_heatmap_path = f'{args.save_dir}/layer_sparsity_heatmap.png'
            if os.path.exists(layer_heatmap_path):
                wandb.log({"layer_sparsity_heatmap": wandb.Image(layer_heatmap_path)})
        except Exception as e:
            print(f"⚠️  Could not upload plots to W&B: {e}")

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
    print(f"KL Budget:     {summary['kl_budget']:.4f}")
    print(f"Final KL:      {summary['final_kl']:.4f} {'✅ WITHIN' if summary['kl_within_budget'] else '⚠️  EXCEEDS'}")
    print(f"Final Accuracy: {summary['final_accuracy']:.4f}")
    print(f"Final Sparsity: {summary['final_sparsity']:.4f}")
    print(f"Best Sparsity:  {summary['best_sparsity']:.4f} (at KL={summary['best_kl']:.4f}, acc={summary['best_accuracy']:.4f})")
    print(f"Total Epochs:   {summary['total_epochs']}")
    print(f"{'='*80}\n")

    # Log final summary to W&B
    if args.wandb:
        try:
            import wandb
            wandb.log({
                'final/kl': summary['final_kl'],
                'final/accuracy': summary['final_accuracy'],
                'final/sparsity': summary['final_sparsity'],
                'final/kl_within_budget': summary['kl_within_budget'],
                'best/sparsity': summary['best_sparsity'],
                'best/accuracy': summary['best_accuracy'],
                'best/kl': summary['best_kl'],
                'best/lambda': summary['best_lambda'],
                'total_epochs': summary['total_epochs'],
            })
            wandb.finish()
        except Exception as e:
            print(f"⚠️  Could not log final summary to W&B: {e}")
