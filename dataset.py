from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from utils.transforms import custom_greyscale_to_tensor
from torch.utils.data import ConcatDataset
import numpy as np

np.random.seed(5241)


class PairedDataset(Dataset):
    def __init__(self, config, dset_A_name, dset_B_name, dset_A_tf, dset_B_tf):
        assert(dset_A_name is not None or dset_A_name != '')
        assert(dset_B_name is not None or dset_B_name != '')

        if dset_A_name == 'MNIST':
            mnist_train = datasets.MNIST('./data', train=True, download=True, transform=dset_A_tf)
            mnist_test = datasets.MNIST('./data', train=False, download=True, transform=dset_A_tf)
            self.dset_A = ConcatDataset([mnist_train, mnist_test])

        if dset_B_name == 'SVHN':
            self.dset_B = datasets.SVHN('./data', download=True, transform=dset_B_tf)

        assert(self.dset_A is not None)
        assert(self.dset_B is not None)

        # Load Dataset B data
        # This loads the whole dataset because we need to group them by label
        dset_B_loader = DataLoader(self.dset_B, batch_size=1000, shuffle=True)
        dset_B_batch = next(iter(dset_B_loader))
        self.dset_B_X, self.dset_B_y = dset_B_batch

        # Group samples per label
        self.dset_B_class_indices = {}
        for idx, label in enumerate(self.dset_B_y):
            y = label.item()
            self.dset_B_class_indices.setdefault(y, []).append(idx)

        print("Dataset B before trim")
        for key in sorted(self.dset_B_class_indices):
            print("class %s size: %d" % (key, len(self.dset_B_class_indices[key])))

        # Select num_dest_per_class samples from dataset B
        for label in self.dset_B_class_indices:
            self.dset_B_class_indices[label] = np.random.choice(self.dset_B_class_indices[label], config.num_dest_per_class)

        print("Dataset B after trim")
        for key in sorted(self.dset_B_class_indices):
            print("class %s size: %d" % (key, len(self.dset_B_class_indices[key])))

    def __len__(self):
        return len(self.dset_A)

    def __getitem__(self, idx):
        # Get Dataset A sample
        dset_A_sample = self.dset_A[idx]
        dset_A_x, dset_A_y = dset_A_sample

        # Get Dataset B sample to pair with Dataset A sample
        dset_B_idx = np.random.choice(self.dset_B_class_indices[dset_A_y], 1)[0]
        dset_B_x = self.dset_B_X[dset_B_idx]

        return dset_A_x, dset_B_x, dset_A_y


class MNIST_SVHN(Dataset):
    def __init__(self, config):
        mnist_transforms = transforms.Compose([
            transforms.RandomCrop(config.rand_crop_sz),
            transforms.Resize(config.input_sz),
            transforms.ToTensor(),
        ])
        svhn_transforms = transforms.Compose([
            transforms.CenterCrop(config.rand_crop_sz),
            transforms.Resize(config.input_sz),
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
