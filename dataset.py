from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import numpy as np

np.random.seed(5241)


class MNIST_SVHN(Dataset):
    def __init__(self, K=10):
        mnist_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        svhn_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Load MNIST data
        self.mnist_data = datasets.MNIST('./data', download=True, transform=mnist_transforms)

        # Load 1k samples from SVHN
        svhn_data = datasets.SVHN('./data', download=True, transform=svhn_transforms)
        svhn_loader = DataLoader(svhn_data, batch_size=1000, shuffle=True)
        svhn_batch = next(iter(svhn_loader))
        self.svhn_x, self.svhn_y = svhn_batch

        # Group samples per label
        self.svhn_class_indices = {}
        for idx, label in enumerate(self.svhn_y):
            y = label.item()
            self.svhn_class_indices.setdefault(y, []).append(idx)

        # Select K samples from svhn
        for label in self.svhn_class_indices:
            self.svhn_class_indices[label] = np.random.choice(self.svhn_class_indices[label], K)

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # Get MNIST Sample
        mnist_sample = self.mnist_data[idx]
        mnist_x, mnist_y = mnist_sample

        # Get SVHN sample to pair with MNIST sample
        svhn_idx = np.random.choice(self.svhn_class_indices[mnist_y], 1)[0]
        svhn_x = self.svhn_x[svhn_idx]

        return mnist_x, svhn_x, mnist_y
