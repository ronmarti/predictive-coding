import logging

import torch
import torch.nn.functional as F

from application.logic.mnist_dataset import MnistDataset
from application.models.pc_network import PcNetwork

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates the PC network on the full MNIST test set.

    Uses forward-only inference (no weight update).
    """

    def __init__(
        self, network: PcNetwork, dataset: MnistDataset, device: torch.device
    ) -> None:
        self._network = network
        self._dataset = dataset
        self._device = device

    def evaluate(self) -> tuple[float, float]:
        """Return (accuracy, avg_cross_entropy_loss) over the test set."""
        correct = 0
        total = 0
        total_loss = 0.0

        for images, labels in self._dataset.test_loader():
            images = images.view(images.size(0), -1).to(self._device)
            labels = labels.to(self._device)

            with torch.no_grad():
                logits = self._network.predict(images)

            loss = F.cross_entropy(logits, labels, reduction="sum")
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return float(accuracy), float(avg_loss)
