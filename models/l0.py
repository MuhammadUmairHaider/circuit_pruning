import torch
import torch.nn as nn
import torch.nn.functional as F

class HardConcreteGate(nn.Module):
    """
    A gate that uses the Hard Concrete distribution to learn binary decisions.
    Based on "Learning Sparse Neural Networks through L0 Regularization" by Louizos et al.
    
    Includes Straight-Through Estimator (STE) for training stability, ensuring
    the model learns to handle binary (0/1) signals during training rather than
    relying on partial "dimmed" signals.
    """
    
    def __init__(
        self, 
        size: int, 
        beta: float = 2.0/3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        # Initialize slightly positive to start with open gates
        init_min=2.5, 
        init_max=3.5
    ):
        """
        Args:
            size: Number of gates
            beta: Temperature parameter (default 2/3 as per paper)
            gamma: Lower stretch parameter (default -0.1)
            zeta: Upper stretch parameter (default 1.1)
            init_min: Minimum value for log_alpha initialization
            init_max: Maximum value for log_alpha initialization
        """
        super().__init__()
        
        # Register buffers for distribution parameters
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("zeta", torch.tensor(zeta))
        
        # Flag for final hard pruning mode
        self.final_mode = False
        
        # Learnable parameters
        self.log_alpha = nn.Parameter(torch.Tensor(size))
        
        # Initialize
        self.init_weights(init_min, init_max)
        
    def init_weights(self, init_min: float, init_max: float):
        """Initialize log_alpha parameters uniformly."""
        with torch.no_grad():
            self.log_alpha.uniform_(init_min, init_max)
    
    def forward(self) -> torch.Tensor:
        # 1. Training with Noise
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-8, 1.0 - 1e-8)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            gate_soft = F.hardtanh(s_stretched, min_val=0, max_val=1)
            
            # STE: Binary 0/1 during training
            gate_hard = (gate_soft > 0.5).float()
            return (gate_hard - gate_soft).detach() + gate_soft

        # 2. Final Mode / Eval (Deterministic)
        else:
            # Remove noise (use expectation)
            s = torch.sigmoid(self.log_alpha)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            gate_soft = F.hardtanh(s_stretched, min_val=0, max_val=1)
            
            if self.final_mode:
                # MATCH TRAINING: Use the exact same threshold logic (Value > 0.5)
                # This is equivalent to checking if log_alpha > 0
                return (gate_soft > 0.5).float()
            else:
                return (gate_soft > 0.5).float()

    def num_gates(self) -> int:
        """Return the number of independent gate logits."""
        return int(self.log_alpha.numel())

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Calculates the expected L0 norm (probability of gate being open).
        
        CRITICAL CHANGE: Returns the MEAN (density) instead of SUM (count).
        This makes the loss scale-invariant to model size.
        """
        p_open = torch.sigmoid(
            self.log_alpha - self.beta * torch.log(-self.gamma / self.zeta)
        )
        return p_open.mean()
    
    def get_sparsity_rate(self) -> float:
        """Returns the expected sparsity rate (fraction of gates that are zero)."""
        p_open = torch.sigmoid(
            self.log_alpha - self.beta * torch.log(-self.gamma / self.zeta)
        )
        return 1.0 - p_open.mean().item()
    
    def get_num_active(self) -> int:
        """Returns the number of active (non-zero) gates if we were to finalize now."""
        with torch.no_grad():
            cutoff = self.beta * torch.log(-self.gamma / self.zeta)
            return (self.log_alpha > cutoff).sum().item()
    
    def set_final_mode(self, mode: bool = True):
        """Enable/disable final hard pruning mode."""
        self.final_mode = mode
    
    def get_mask_statistics(self) -> dict:
        """Get detailed statistics about the gates."""
        with torch.no_grad():
            # Use the deterministic evaluation view for stats
            s = torch.sigmoid(self.log_alpha)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
            
            stats = {
                'mean_gate': gate.mean().item(),
                'std_gate': gate.std().item(),
                'min_gate': gate.min().item(),
                'max_gate': gate.max().item(),
                'sparsity_rate': self.get_sparsity_rate(),
                'num_active': self.get_num_active(),
                'num_total': gate.numel(),
                'expected_density': self.get_sparsity_loss().item()
            }
            
            # Add percentiles
            for p in [10, 25, 50, 75, 90]:
                stats[f'percentile_{p}'] = torch.quantile(gate, p/100.0).item()
                
            return stats