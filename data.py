import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10

class CIFAR10Data():
    def __init__(self, data_dir, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self.mean, self.std),
            ]
        )

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.trainset = CIFAR10(data_dir, train=True, transform=transform, download=True)
        self.testset  = CIFAR10(data_dir, train=False, transform=transform, download=True)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()
