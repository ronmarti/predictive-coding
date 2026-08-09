import torch

from application.models.pc_layer import PcLayer


class AdamOptimizer:
    """Adam optimiser operating on PcLayer.grad dicts (no nn.Parameter).

    Maintains first and second moment tensors per layer per parameter.
    """

    def __init__(self, lr: float = 1e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._t = 0
        # Keyed by layer id -> param name -> moment tensor
        self._m: dict[int, dict[str, torch.Tensor]] = {}
        self._v: dict[int, dict[str, torch.Tensor]] = {}

    def reset(self) -> None:
        """Clear accumulated moment estimates so a fresh network starts clean."""
        self._t = 0
        self._m = {}
        self._v = {}

    def step(self, layers: list[PcLayer]) -> None:
        """Apply one Adam update step to all layers with accumulated grads."""
        self._t += 1
        bc1 = 1.0 - self._beta1 ** self._t
        bc2 = 1.0 - self._beta2 ** self._t

        for layer in layers:
            lid = id(layer)
            if lid not in self._m:
                self._m[lid] = {}
                self._v[lid] = {}

            for param_name in ("weights", "biases"):
                if param_name not in layer.grad:
                    continue
                g = layer.grad[param_name]
                if param_name not in self._m[lid]:
                    self._m[lid][param_name] = torch.zeros_like(g)
                    self._v[lid][param_name] = torch.zeros_like(g)

                m = self._m[lid][param_name]
                v = self._v[lid][param_name]
                m.mul_(self._beta1).add_(g, alpha=1.0 - self._beta1)
                v.mul_(self._beta2).addcmul_(g, g, value=1.0 - self._beta2)

                m_hat = m / bc1
                v_hat = v / bc2
                update = m_hat / (v_hat.sqrt() + self._eps)

                param = getattr(layer, param_name)
                param.add_(update, alpha=self._lr)
