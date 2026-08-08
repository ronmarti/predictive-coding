import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """All runtime configuration read from environment variables."""

    nodes: list[int] = field(default_factory=lambda: [784, 300, 100, 10])
    mu_dt: float = 0.01
    n_infer_iters: int = 50
    lr: float = 1e-4
    activation: str = "relu"
    eval_interval: int = 100
    auto_batch_size: int = 64
    device: str = "cuda"

    @classmethod
    def from_env(cls) -> "Config":
        """Construct Config from environment variables, falling back to defaults."""
        raw_nodes = os.environ.get("NODES", "784,300,100,10")
        nodes = [int(n) for n in raw_nodes.split(",")]
        return cls(
            nodes=nodes,
            mu_dt=float(os.environ.get("MU_DT", "0.01")),
            n_infer_iters=int(os.environ.get("N_INFER_ITERS", "50")),
            lr=float(os.environ.get("LR", "1e-4")),
            activation=os.environ.get("ACTIVATION", "relu"),
            eval_interval=int(os.environ.get("EVAL_INTERVAL", "100")),
            auto_batch_size=int(os.environ.get("AUTO_BATCH_SIZE", "64")),
            device=os.environ.get("DEVICE", "cuda"),
        )
