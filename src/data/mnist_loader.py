from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_datasets(data_dir: Path):
    """Download/load MNIST train and test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def make_loader(dataset, batch_size: int = 128, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
