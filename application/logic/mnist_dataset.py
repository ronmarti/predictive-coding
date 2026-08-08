import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class MnistDataset:
    """Wraps torchvision MNIST for random sampling and test iteration.

    Downloads to TORCH_HOME/datasets (cached across container restarts
    via the mnist-cache Docker volume).
    """

    _MEAN = 0.1307
    _STD = 0.3081

    def __init__(self, device: torch.device) -> None:
        self._device = device
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self._MEAN,), (self._STD,)),
        ])
        cache_dir = os.path.join(
            torch.hub.get_dir(), "datasets"
        )
        self._train = torchvision.datasets.MNIST(
            root=cache_dir, train=True, download=True, transform=transform
        )
        self._test = torchvision.datasets.MNIST(
            root=cache_dir, train=False, download=True, transform=transform
        )
        self._train_indices = list(range(len(self._train)))

    # ------------------------------------------------------------------
    # Training samples
    # ------------------------------------------------------------------

    def random_sample(self) -> tuple[torch.Tensor, int]:
        """Return a random (normalised flat tensor, label) from train set."""
        idx = random.randint(0, len(self._train) - 1)
        img_tensor, label = self._train[idx]
        return img_tensor.view(1, -1).to(self._device), int(label)

    def random_batch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (batch, 784) tensor and (batch,) label tensor."""
        indices = [
            random.randint(0, len(self._train) - 1)
            for _ in range(batch_size)
        ]
        imgs = [self._train[i][0].view(-1) for i in indices]
        labels = [self._train[i][1] for i in indices]
        return (
            torch.stack(imgs).to(self._device),
            torch.tensor(labels, device=self._device),
        )

    def raw_image(self, idx: int) -> np.ndarray:
        """Return raw (unnormalised) 28x28 numpy array for display."""
        img_tensor, _ = self._train[idx]
        return img_tensor.squeeze().numpy()

    def random_raw(self) -> tuple[np.ndarray, torch.Tensor, int]:
        """Return (raw 28x28, normalised flat tensor, label) simultaneously."""
        idx = random.randint(0, len(self._train) - 1)
        raw_img, label = self._train.data[idx].numpy(), \
            int(self._train.targets[idx])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self._MEAN,), (self._STD,)),
        ])
        img_tensor = transform(
            self._train.data[idx].numpy()
        ).view(1, -1).to(self._device)
        return raw_img, img_tensor, label

    # ------------------------------------------------------------------
    # Test samples
    # ------------------------------------------------------------------

    def test_loader(
        self, batch_size: int = 256
    ) -> torch.utils.data.DataLoader:
        """Return a DataLoader over the full MNIST test set."""
        return torch.utils.data.DataLoader(
            self._test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
