import torchvision
import torchvision.transforms as tf
import torch.utils.data
import PIL.Image

class MNISTDataset(torch.utils.data.Dataset):
    """
    Define a MNIST dataset (and variant) that return the data, targt and index.
    """
    def __init__(self, dataset_name, data_path, train=True, data_augmentation=True):
        """
        Build a MNIST dataset.
        ----------
        INPUT
            |---- dataset_name (str) the dataset to load : only  MNIST,
            |           FashionMNIST, KMNIST or QMNIST are supported.
            |---- data_path (str) where to find the data.
            |---- train (bool) whether the train set is used.
            |---- data_augmentation (bool) whether data_augmentation is performed
            |           (random rotation, scale, translation and brighness changes)
        """
        if data_augmentation:
            transform = tf.Compose([tf.RandomAffine(15, translate=(0.05, 0.05),
                                                    scale=(0.8,1.2),
                                                    resample=PIL.Image.BILINEAR),
                                    tf.ColorJitter(brightness=(0.8,1.2)),
                                    tf.ToTensor()])
        else:
            transform = tf.ToTensor()

        if dataset_name == 'MNIST':
            self.dataset = torchvision.datasets.MNIST(data_path, train=train,
                                              download=True, transform=transform)
        elif dataset_name == 'FashionMNIST':
            self.dataset =  torchvision.datasets.FashionMNIST(data_path, train=train,
                                              download=True, transform=transform)
        elif dataset_name == 'KMNIST':
            self.dataset = torchvision.datasets.KMNIST(data_path, train=train,
                                              download=True, transform=transform)
        elif dataset_name == 'QMNIST':
            self.dataset = torchvision.datasets.QMNIST(data_path, train=train,
                                              download=True, transform=transform)
        else:
            raise ValueError('Non-supported dataset name')

    def __len__(self):
        """
        Return the len of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Redefine getitem function to recover indices.
        """
        data, target = self.dataset[index]
        return data, target, index
