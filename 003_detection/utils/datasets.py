from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from funcdispatch import funcdispatch

class dataset():

    # パスを指定してデータを取得
    @funcdispatch()
    def loadDataloader(self, 
                       path, 
                       transform= transforms.Compose([transforms.Resize(256), transforms.ToTensor()]),
                       batch_size=3):
        dataset = ImageFolder(path, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return dataloader

    # 一つのフォルダに格納されているデータをtrain, valid, testに分ける
    @loadDataloader.register
    def _(self, 
          path, 
          transform= transforms.Compose([transforms.Resize(256), transforms.ToTensor()]),
          batch_size=3,
          shuffle=True,
          num_workers=1,
          train=0.6,
          valid=0.2,
          test=0.2):
        # 画像読み込み
        dataset = ImageFolder(path, transform)

        # データセットをtrainとtestに分割
        train_indices, test_indices = train_test_split(list(range(len(dataset.targets))), test_size=test, stratify=dataset.targets)
        _train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        # データセットをtrainとvalidationに分割
        train_indices, val_indices = train_test_split(list(range(len(_train_dataset.targets))), test_size=valid, stratify=_train_dataset.targets)
        train_dataset = torch.utils.data.Subset(_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(_train_dataset, val_indices)

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle, 
                                      num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle, 
                                      num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle, 
                                      num_workers=num_workers)

        dataloader = {
            'train': train_dataloader,
            'valid': valid_dataloader,
            'test': test_dataloader
        }

        return dataloader


    # train, valid, testフォルダごとにデータを取得
    @loadDataloader.register
    def _(self, 
          train_path, 
          valid_path,
          test_path,
          transform= transforms.Compose([transforms.Resize(256), transforms.ToTensor()]),
          batch_size=3,
          shuffle=True,
          num_workers=1):
        # 画像読み込み
        train_dataset = ImageFolder(train_path, transform)
        valid_dataset = ImageFolder(valid_path, transform)
        test_dataset = ImageFolder(test_path, transform)

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle, 
                                      num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle, 
                                      num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle, 
                                      num_workers=num_workers)

        dataloader = {
            'train': train_dataloader,
            'valid': valid_dataloader,
            'test': test_dataloader
        }

        return dataloader
