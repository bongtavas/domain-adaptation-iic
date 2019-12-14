from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from utils.transforms import custom_greyscale_to_tensor
import numpy as np

np.random.seed(5241)


class MNIST_SVHN(Dataset):
    def __init__(self, config):
        mnist_transforms = transforms.Compose([
            transforms.RandomCrop(config.rand_crop_sz),
            transforms.Resize(config.input_sz),
            transforms.ToTensor(),
        ])
        svhn_transforms = transforms.Compose([
            transforms.CenterCrop(config.input_sz),
            custom_greyscale_to_tensor(config.include_rgb),
        ])

        # Load MNIST data
        self.mnist_data = datasets.MNIST('./data', download=True, transform=mnist_transforms)
        
        # Load SVHN data
        self.svhn_data = datasets.SVHN('./data', download=True, transform=svhn_transforms)
        svhn_loader = DataLoader(self.svhn_data, batch_size=len(self.svhn_data), shuffle=True)
        svhn_batch = next(iter(svhn_loader))
        self.svhn_x, self.svhn_y = svhn_batch

        # Group samples per label
        self.svhn_class_indices = {}
        for idx, label in enumerate(self.svhn_y):
            y = label.item()
            self.svhn_class_indices.setdefault(y, []).append(idx)

        print("SVHN dataset before trim")
        for key in sorted(self.svhn_class_indices):
            print("class %s size: %d" % (key, len(self.svhn_class_indices[key])))

        # Select num_dest_per_class samples from svhn
        for label in self.svhn_class_indices:
            self.svhn_class_indices[label] = np.random.choice(self.svhn_class_indices[label], config.num_dest_per_class)

        print("SVHN dataset after trim")
        for key in sorted(self.svhn_class_indices):
            print("class %s size: %d" % (key, len(self.svhn_class_indices[key])))

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
