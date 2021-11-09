import torch
from torchvision import datasets, transforms
import numpy as np

def loadRandData():
    """
    ランダムな数字の生成

    Parameters
    ----------

    Returns
    -------
    X_train : list
        入力値（学習）
    y_train : list
        出力値（学習）
    X_test : list
        入力値（テスト）
    """

    # ----- 学習データ作成 -----
    x = np.random.uniform(0, 10, 100)
    y = np.random.uniform(0.2, 1.9, 100) + x + 10

    # テンソルに変換
    x = torch.from_numpy(x.astype(np.float32)).float()
    y_train = torch.from_numpy(y.astype(np.float32)).float()
    # xに切片用の定数1配列を結合
    X_train = torch.stack([torch.ones(100), x], 1)



    # ----- テストデータ作成 -----
    x_test = np.linspace(-5, 15, 15)
    x_test = torch.from_numpy(x_test.astype(np.float32)).float()
    X_test = torch.stack([torch.ones(15), x_test], 1)

    return X_train, y_train, X_test

def loadMNIST(batch=128, intensity=1.0):
    """
    MNISTのデータ取得

    Parameters
    ----------
    batch : int
        バッチサイズ
    intensity : float
        

    Returns
    -------
    """

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)

    return {'train': train_loader, 'test': test_loader}
