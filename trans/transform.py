__metaclass__ = type

from torchvision import transforms
from transform_constants import *

class TransformConstructor:
    def construct(self, phase, dataset_name, obj):
        targetClass = getattr(self, dataset_name)
        instance = targetClass(obj)

        if phase == 'train':
            return instance.train_transform()
        elif (phase == 'test') or (phase == 'analysis'):
            return instance.test_transform()
        else:
            raise Exception('Wrong phase!')

    class Dataset():
        def __init__(self, obj):
            self.obj = obj
            self.pad = int((self.obj.padded_im_size - self.obj.im_size)/2)
            self.mean = ()
            self.std = ()
            for _ in range(self.obj.input_ch):
                self.mean += (0.5,)
                self.std  += (0.5,)

        def train_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

        def test_transform(self):
            return self.train_transform()


    class MNIST(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.MNIST, self).__init__(obj)
            self.mean = (MNIST_MEAN,)
            self.std = (MNIST_STD,)

    class FashionMNIST(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.FashionMNIST, self).__init__(obj)
            self.mean = (FASHION_MNIST_MEAN,)
            self.std = (FASHION_MNIST_STD,)

    class CIFAR10(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.CIFAR10, self).__init__(obj)
            self.mean = [x / 255 for x in CIFAR10_MEAN]
            self.std = [x / 255 for x in CIFAR10_STD]

        def train_transform(self):
            return transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

        def test_transform(self):
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

    class CIFAR100(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.CIFAR100, self).__init__(obj)
            self.mean = [x / 255 for x in CIFAR100_MEAN]
            self.std = [x / 255 for x in CIFAR100_STD]

        def train_transform(self):

            return transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

        def test_transform(self):
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

    class STL10(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.STL10, self).__init__(obj)
            self.mean = STL10_MEAN
            self.std = STL10_STD

        def train_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

        def test_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

    class SVHN(Dataset):
        def __init__(self, obj):
            super(TransformConstructor.SVHN, self).__init__(obj)
            self.mean = SVHN_MEAN
            self.std = SVHN_STD

        def train_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

        def test_transform(self):
            return transforms.Compose([
                    transforms.Pad(self.pad),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])
