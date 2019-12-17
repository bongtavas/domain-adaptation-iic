from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.utils.data import ConcatDataset
import numpy as np

np.random.seed(5241)


class TestDataset(Dataset):

    def __init__(self, config, dset_name, dset_tf):
        if dset_name == 'MNIST':
            self.dset = datasets.MNIST('./data', train=False, download=True, transform=dset_tf)
        elif dset_name == 'SVHN':
            self.dset = datasets.SVHN('./data', split='test', download=True, transform=dset_tf)
        if dset_name == 'Fashion_MNIST':
            self.dset = datasets.FashionMNIST('./data', download=True, transform=dset_tf)
        elif dset_name == 'Fashion_WILD':
            self.dset = datasets.ImageFolder('./data/FASHION_WILD/test', transform=dset_tf)

        assert(self.dset is not None)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        return self.dset[idx]


class PairedDataset(Dataset):
    def __init__(self, config, dset_A_name, dset_B_name, dset_A_tf, dset_B_tf):
        assert(dset_A_name is not None or dset_A_name != '')
        assert(dset_B_name is not None or dset_B_name != '')

        if dset_A_name == 'MNIST':
            mnist_train = datasets.MNIST('./data', train=True, download=True, transform=dset_A_tf)
            mnist_test = datasets.MNIST('./data', train=False, download=True, transform=dset_A_tf)
            self.dset_A = ConcatDataset([mnist_train, mnist_test])
        elif dset_A_name == 'Fashion_MNIST':
            fmnist_train = datasets.FashionMNIST('./data', train=True, download=True, transform=dset_A_tf)
            fmnist_test = datasets.FashionMNIST('./data', train=False, download=True, transform=dset_A_tf)
            self.dset_A = ConcatDataset([fmnist_train, fmnist_test])

        if dset_B_name == 'SVHN':
            self.dset_B = datasets.SVHN('./data', download=True, transform=dset_B_tf)
        elif dset_B_name == 'Fashion_WILD':
            self.dset_B = datasets.ImageFolder('./data/FASHION_WILD/train', transform=dset_B_tf)

        assert(self.dset_A is not None)
        assert(self.dset_B is not None)

        # Load Dataset B data

        if config.dset_B_all:
            dset_B_sz = len(self.dset_B)
        else:
            dset_B_sz = 1000

        dset_B_loader = DataLoader(self.dset_B, batch_size=dset_B_sz, shuffle=True)
        dset_B_batch = next(iter(dset_B_loader))
        self.dset_B_X, self.dset_B_y = dset_B_batch

        # Group samples per label
        self.dset_B_class_indices = {}
        for idx, label in enumerate(self.dset_B_y):
            y = label.item()
            self.dset_B_class_indices.setdefault(y, []).append(idx)

        if not config.dset_B_all:
            # Select num_dest_per_class samples from dataset B
            for label in self.dset_B_class_indices:
                self.dset_B_class_indices[label] = np.random.choice(self.dset_B_class_indices[label], config.num_dest_per_class)
            print("Dataset B after trim")
        else:
            print("Using all samples in dataset B")

        for key in sorted(self.dset_B_class_indices):
            print("class %s size: %d" % (key, len(self.dset_B_class_indices[key])))

        print("Dataset A size: %d" % len(self.dset_A))
        print("Dataset B size: %d" % len(self.dset_B))

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
