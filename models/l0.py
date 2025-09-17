import torch
import torch.nn as nn
import torch.nn.functional as F

class HardConcreteGate(nn.Module):
    """
    A gate that uses the Hard Concrete distribution to learn binary decisions.
    Based on "Learning Sparse Neural Networks through L0 Regularization" by Louizos et al.
    
    The Hard Concrete distribution is a stretched and clipped sigmoid that allows
    for exact zeros while maintaining differentiability.
    """
    
    def __init__(
        self, 
        size: int, 
        beta: float = 2.0/3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        # init_min: float = 0.1, 
        # init_max: float = 1.1
        init_min=1.5, 
        init_max=2.5
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
        """
        Samples from the Hard Concrete distribution.
        
        Returns:
            Gate values in [0, 1]
            - Training: Stochastic samples from Hard Concrete
            - Eval: Deterministic gates based on expected value
            - Final mode: Hard binary decisions (0 or 1)
        """
        if self.final_mode:
            # Hard binary decisions for final circuit
            s = torch.sigmoid(self.log_alpha)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
            return (gate > 0.5).float()
        
        if self.training:
            # Sample from Hard Concrete during training
            u = torch.rand_like(self.log_alpha)
            u = u.clamp(1e-8, 1.0 - 1e-8)  # Numerical stability
            
            # Reparameterization trick with Gumbel-Softmax
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
            )
        else:
            # Expected value during evaluation
            s = torch.sigmoid(self.log_alpha)
        
        # Stretch and clip
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
        
        return gate
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Calculates the expected L0 norm under the Hard Concrete distribution.
        
        This is the expected number of non-zero gates.
        """
        # Probability of gate being non-zero
        p_open = torch.sigmoid(
            self.log_alpha - self.beta * torch.log(-self.gamma / self.zeta)
        )
        return p_open.sum()
    
    def get_sparsity_rate(self) -> float:
        """Returns the expected sparsity rate (fraction of gates that are zero)."""
        p_open = torch.sigmoid(
            self.log_alpha - self.beta * torch.log(-self.gamma / self.zeta)
        )
        return 1.0 - p_open.mean().item()
    
    def get_num_active(self) -> int:
        """Returns the number of active (non-zero) gates in final mode."""
        with torch.no_grad():
            s = torch.sigmoid(self.log_alpha)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
            return (gate > 0.5).sum().item()
    
    def set_final_mode(self, mode: bool = True):
        """Enable/disable final hard pruning mode."""
        self.final_mode = mode
    
    def get_mask_statistics(self) -> dict:
        """Get detailed statistics about the gates."""
        with torch.no_grad():
            s = torch.sigmoid(self.log_alpha)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
            
            stats = {
                'mean_gate': gate.mean().item(),
                'std_gate': gate.std().item(),
                'min_gate': gate.min().item(),
                'max_gate': gate.max().item(),
                'sparsity_rate': self.get_sparsity_rate(),
                'num_active': (gate > 0.5).sum().item(),
                'num_total': gate.numel(),
                'expected_l0': self.get_sparsity_loss().item()
            }
            
            # Add percentiles
            for p in [10, 25, 50, 75, 90]:
                stats[f'percentile_{p}'] = torch.quantile(gate, p/100.0).item()
                
            return stats