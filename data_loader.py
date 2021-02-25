import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets

class DataSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def get_train_data_loaders(self):
        train_dataset = datasets.CIFAR100('./data', train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              self.normalize
                                          ]))
        return self.get_train_validation_data_loaders(train_dataset)

    def get_train_validation_data_loaders(self, train_dataset):
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader

    def get_test_loaders(self):
        test_datasets = datasets.CIFAR100('./data', train=False, download=True, 
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              self.normalize
                                          ]))
        return DataLoader(test_datasets, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=False)