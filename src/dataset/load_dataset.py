import torchvision
import torchvision.transforms as tf
import PIL.Image

def load_dataset(dataset_name, data_path, train=True, data_augmentation=True):
    """
    load the dataset specified by `dataset_name`.
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
        return torchvision.datasets.MNIST(data_path, train=train,
                                          download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(data_path, train=train,
                                          download=True, transform=transoform)
    elif dataset_name == 'KMNIST':
        return torchvision.datasets.KMNIST(data_path, train=train,
                                          download=True, transform=transoform)
    elif dataset_name == 'QMNIST':
        return torchvision.datasets.QMNIST(data_path, train=train,
                                          download=True, transform=transoform)
    else:
        raise ValueError('Non-supported dataset name')
