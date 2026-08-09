import torch
import torch.nn.functional as F

from application.models.activations import Activation, LinearActivation
from application.models.pc_layer import PcLayer
from application.models.pc_optimizer import AdamOptimizer


class PcNetwork:
    """Hierarchical predictive coding network of fully-connected layers.

    Architecture is defined by a list of node counts, e.g.
    [784, 300, 100, 10] creates three PcLayers.
    """

    def __init__(
        self,
        nodes: list[int],
        hidden_activation: Activation,
        device: torch.device,
    ) -> None:
        if len(nodes) < 2:
            raise ValueError("nodes must have at least 2 elements.")
        self._device = device
        self._nodes = nodes
        # Output layer always has linear activation
        self.layers: list[PcLayer] = [
            PcLayer(
                nodes[i],
                nodes[i + 1],
                hidden_activation if i < len(nodes) - 2 else LinearActivation(),
                device,
            )
            for i in range(len(nodes) - 1)
        ]

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _init_mu(
        self, inp: torch.Tensor
    ) -> list[torch.Tensor]:
        """Forward-pass initialisation of latent states mu."""
        mu = [inp]
        x = inp
        for layer in self.layers:
            x = layer.forward(x)
            mu.append(x)
        return mu

    def run_inference(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        n_iters: int,
        mu_dt: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Iterative inference: minimise free energy by gradient descent on mu.

        inp and target are clamped; only intermediate mu layers are updated.
        Returns (mu, errors) after convergence.
        """
        mu = self._init_mu(inp)
        mu[-1] = target.clone()
        errors: list[torch.Tensor] = [torch.zeros_like(m) for m in mu[1:]]

        for _ in range(n_iters):
            # Compute predictions and errors bottom-up
            for i, layer in enumerate(self.layers):
                pred = layer.forward(mu[i])
                errors[i] = mu[i + 1] - pred

            # Update only intermediate mu (not input or target)
            for i in range(1, len(mu) - 1):
                err_from_above = self.layers[i].backward(errors[i])
                mu[i] = mu[i] - mu_dt * (-err_from_above + errors[i - 1])

        # Final forward pass so _inp/_pre_act are fresh for gradient accumulation
        for i, layer in enumerate(self.layers):
            pred = layer.forward(mu[i])
            errors[i] = mu[i + 1] - pred

        return mu, errors

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(
        self,
        errors: list[torch.Tensor],
        optimizer: AdamOptimizer,
        norm_factor: int = 1,
    ) -> None:
        """Accumulate and apply Hebbian weight updates via optimizer."""
        for layer in self.layers:
            layer.zero_grad()
        for i, layer in enumerate(self.layers):
            layer.update_gradient(errors[i], norm_factor)
        optimizer.step(self.layers)

    # ------------------------------------------------------------------
    # Inference-only prediction
    # ------------------------------------------------------------------

    def predict(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass only — returns raw logits (no weight update)."""
        x = inp
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def confidence(self, inp: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for all 10 classes."""
        logits = self.predict(inp)
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Device and persistence
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "PcNetwork":
        """Move all layer tensors to device."""
        self._device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def reset(self) -> None:
        """Reinitialise all layer weights and biases to their initial distribution."""
        for layer in self.layers:
            layer.reset()

    def state_dict(self) -> list[dict]:
        """Return serialisable state for all layers."""
        return [layer.state_dict() for layer in self.layers]

    def load_state_dict(self, states: list[dict]) -> None:
        """Restore all layers from state dicts."""
        if len(states) != len(self.layers):
            raise ValueError(
                f"State has {len(states)} layers, "
                f"network has {len(self.layers)}."
            )
        for layer, state in zip(self.layers, states):
            layer.load_state_dict(state)
