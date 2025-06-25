import torch
import torch.nn as nn
import torch.nn.functional as F

class HardConcreteGate(nn.Module):
    """
    A gate that uses the Hard Concrete distribution to learn binary decisions.
    Now includes a `final_mode` for evaluating the hard-pruned circuit.
    """
    def __init__(self, size: int, temperature: float = 0.5, init_min: float = 0.1, init_max: float = 0.1):
        super().__init__()

        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("gamma", torch.tensor(-0.1))
        self.register_buffer("zeta", torch.tensor(1.1))
        self.register_buffer("beta", torch.tensor(2.0 / 3.0))
        
        # --- NEW: A flag to control the final circuit mode ---
        self.final_mode = False

        self.log_alpha = nn.Parameter(torch.Tensor(size))
        self.init_weights(init_min, init_max)

    def init_weights(self, init_min, init_max):
        with torch.no_grad():
            self.log_alpha.uniform_(init_min, init_max)

    def forward(self) -> torch.Tensor:
        """
        Samples from the Hard Concrete distribution.
        - In `final_mode`, it's a hard 0/1 decision.
        - In `train` mode, it's a stochastic sample.
        - In `eval` mode, it's a deterministic "soft" gate value.
        """
        # --- NEW: Check for the final circuit mode first ---
        if self.final_mode:
            # In final mode, we make a hard decision based on the learned log_alpha
            s = torch.sigmoid(self.log_alpha)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            gate = F.hardtanh(s_stretched, min_val=0, max_val=1)
            # Round to the nearest integer (0 or 1)
            return (gate > 0.5).float()

        # Original logic for training and standard evaluation
        if self.training:
            u = torch.rand(self.log_alpha.size(), device=self.log_alpha.device)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.temperature)
        else:
            s = torch.sigmoid(self.log_alpha)

        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        gate = F.hardtanh(s_stretched, min_val=0, max_val=1)

        return gate

    def get_sparsity_loss(self) -> torch.Tensor:
        """Calculates the sparsity penalty (unchanged)."""
        p_open = torch.sigmoid(self.log_alpha - self.temperature * self.beta * torch.log(-self.gamma / self.zeta))
        return p_open.sum()