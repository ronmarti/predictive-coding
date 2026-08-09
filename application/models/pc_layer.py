import math
from typing import Optional

import torch

from application.models.activations import Activation


class PcLayer:
    """One fully-connected predictive coding layer.

    Holds weights W (out x in) and biases b (out,).
    All gradients are accumulated manually — no autograd.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Activation,
        device: torch.device,
    ) -> None:
        self._device = device
        self._activation = activation
        # Xavier uniform init for stable PC dynamics
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.weights = (
            torch.empty(out_dim, in_dim, device=device)
            .uniform_(-limit, limit)
        )
        self.biases = torch.zeros(out_dim, device=device)
        self.grad: dict[str, torch.Tensor] = {}
        self._inp: Optional[torch.Tensor] = None
        self._pre_act: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward / backward (no autograd)
    # ------------------------------------------------------------------

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Store input and pre-activation; return activation output."""
        self._inp = inp
        self._pre_act = inp @ self.weights.T + self.biases
        return self._activation.forward(self._pre_act)

    def backward(self, err_above: torch.Tensor) -> torch.Tensor:
        """Propagate error downward; return error at this layer's input."""
        deriv = self._activation.deriv(self._pre_act)
        return (err_above * deriv) @ self.weights

    def update_gradient(
        self, err_above: torch.Tensor, norm_factor: int = 1
    ) -> None:
        """Accumulate Hebbian weight/bias gradients from converged errors."""
        deriv = self._activation.deriv(self._pre_act)
        delta = err_above * deriv  # (batch, out)
        if "weights" not in self.grad:
            self.grad["weights"] = torch.zeros_like(self.weights)
            self.grad["biases"] = torch.zeros_like(self.biases)
        # norm_factor keeps the gradient scale consistent across batch sizes
        self.grad["weights"] += (delta.T @ self._inp) / norm_factor
        self.grad["biases"] += delta.sum(dim=0) / norm_factor

    def zero_grad(self) -> None:
        """Reset accumulated gradients."""
        self.grad = {}

    def reset(self) -> None:
        """Reinitialise weights (Xavier uniform) and biases (zero)."""
        in_dim, out_dim = self.weights.shape[1], self.weights.shape[0]
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.weights.uniform_(-limit, limit)
        self.biases.zero_()
        self.grad = {}

    def to(self, device: torch.device) -> "PcLayer":
        """Move weights and biases to device in-place."""
        self._device = device
        self.weights = self.weights.to(device)
        self.biases = self.biases.to(device)
        return self

    def state_dict(self) -> dict:
        """Return serialisable weight state."""
        return {
            "weights": self.weights.cpu(),
            "biases": self.biases.cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore weights from a state dict."""
        self.weights = state["weights"].to(self._device)
        self.biases = state["biases"].to(self._device)
