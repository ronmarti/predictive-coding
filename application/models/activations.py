import abc

import torch


class Activation(abc.ABC):
    """Abstract base for element-wise activation functions."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation to x."""
        ...

    @abc.abstractmethod
    def deriv(self, x: torch.Tensor) -> torch.Tensor:
        """Return element-wise derivative evaluated at x (pre-activation)."""
        ...


class ReLU(Activation):
    """Rectified linear unit activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def deriv(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).to(x.dtype)


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def deriv(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.tanh(x) ** 2


class LinearActivation(Activation):
    """Identity activation — used on the output layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def deriv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


def build_activation(name: str) -> Activation:
    """Instantiate an Activation by name string."""
    mapping = {"relu": ReLU, "tanh": Tanh, "linear": LinearActivation}
    key = name.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(mapping.keys())}"
        )
    return mapping[key]()
