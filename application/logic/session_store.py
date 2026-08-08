import logging
from pathlib import Path
from typing import Optional

import numpy as np
import solara
import torch
import torch.nn.functional as F

from application.logic.mnist_dataset import MnistDataset
from application.models.pc_network import PcNetwork

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reactive application state
# ---------------------------------------------------------------------------

current_image: solara.Reactive[Optional[np.ndarray]] = solara.reactive(None)
predicted_label: solara.Reactive[int] = solara.reactive(-1)
true_label: solara.Reactive[int] = solara.reactive(-1)
confidence_scores: solara.Reactive[list[float]] = solara.reactive(
    [0.0] * 10
)
auto_mode: solara.Reactive[bool] = solara.reactive(False)
samples_seen: solara.Reactive[int] = solara.reactive(0)
# Each entry: {"samples_seen": int, "accuracy": float, "avg_loss": float}
history: solara.Reactive[list[dict]] = solara.reactive([])
status_message: solara.Reactive[str] = solara.reactive("Initialising…")
weights_file_path: solara.Reactive[Optional[Path]] = solara.reactive(None)
is_busy: solara.Reactive[bool] = solara.reactive(False)


class SessionStore:
    """Manages the current MNIST sample and updates reactive state.

    Acts as the single point of contact between the PC network and the UI.
    """

    def __init__(
        self,
        network: PcNetwork,
        dataset: MnistDataset,
        device: torch.device,
        n_infer_iters: int,
        mu_dt: float,
    ) -> None:
        self._network = network
        self._dataset = dataset
        self._device = device
        self._n_infer_iters = n_infer_iters
        self._mu_dt = mu_dt
        # Current flat normalised tensor for the displayed digit
        self._current_tensor: Optional[torch.Tensor] = None

    def advance_sample(self) -> None:
        """Load a new random MNIST sample and run inference on it."""
        raw_img, img_tensor, label = self._dataset.random_raw()
        self._current_tensor = img_tensor
        true_label.value = label

        with torch.no_grad():
            scores = self._network.confidence(img_tensor)

        scores_list = scores.squeeze().cpu().tolist()
        predicted = int(torch.argmax(scores.squeeze()).item())

        current_image.value = raw_img
        confidence_scores.value = scores_list
        predicted_label.value = predicted

    def learn_next_sample(self, optimizer) -> None:
        """Load a fresh random sample and learn from it without updating UI."""
        _, img_tensor, label = self._dataset.random_raw()
        self._current_tensor = img_tensor
        target = self._make_target(label)
        _, errors = self._network.run_inference(
            img_tensor, target, self._n_infer_iters, self._mu_dt
        )
        self._network.update_weights(errors, optimizer)
        samples_seen.value = samples_seen.value + 1

    def learn_batch(self, batch_size: int, optimizer) -> None:
        """Learn from a mini-batch without updating UI reactives."""
        img_tensors, labels = self._dataset.random_batch(batch_size)
        targets = torch.zeros(batch_size, 10, device=self._device)
        targets.scatter_(1, labels.view(-1, 1), 1.0)
        _, errors = self._network.run_inference(
            img_tensors, targets, self._n_infer_iters, self._mu_dt
        )
        self._network.update_weights(errors, optimizer)
        samples_seen.value = samples_seen.value + batch_size

    def learn_current(self, optimizer) -> None:
        """Run inference with clamped target and apply weight update."""
        if self._current_tensor is None:
            return
        target = self._make_target(true_label.value)
        _, errors = self._network.run_inference(
            self._current_tensor,
            target,
            self._n_infer_iters,
            self._mu_dt,
        )
        self._network.update_weights(errors, optimizer)
        samples_seen.value = samples_seen.value + 1

    def learn_with_label(self, label: int, optimizer) -> None:
        """Override true label, learn, then advance to next sample."""
        true_label.value = label
        self.learn_current(optimizer)
        self.advance_sample()

    def skip_current(self) -> None:
        """Advance without learning."""
        self.advance_sample()

    def _make_target(self, label: int) -> torch.Tensor:
        """One-hot target tensor for a given digit label."""
        target = torch.zeros(1, 10, device=self._device)
        target[0, label] = 1.0
        return target
