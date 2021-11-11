##############################
# import
##############################
# 学習、テスト関数
from utils.train import TrainClass
# 出力クラス
from utils.plot import Plot
# モデル
from models.model import MyNet
import torch.nn.functional as f
from torch import optim

# 学習、テストデータ
from utils.datasets import loadMNIST

##############################
# パラメータ設定
##############################
# 学習用クラス
trainClass = TrainClass()
trainClass.fixSeed(seed=2021) # 乱数の固定
hypParams = {
    'in_features': 28*28,
    'out_features': 10,
    'lr': 0.001,
    'iteration': 20
}

# モデルの設定
model = MyNet(in_features=hypParams['in_features'], 
                         out_features=hypParams['out_features'])
# 最適化
optimizer = optim.Adam(model.parameters(), lr=hypParams['lr'])
# 損失関数
E = f.nll_loss

# 出力クラス
plot = Plot()

##############################
# データ取得
##############################
datasets = loadMNIST()


##############################
# 学習
##############################
model, losses = trainClass.train(model=model, 
                       optimizer=optimizer, 
                       E=E, 
                       iteration=hypParams['iteration'], 
                       dataloader=datasets, 
                       valid=True,
                       verbose=1)
                       
##############################
# テスト
##############################
testLoss = trainClass.test(model=model, 
                           E=E, 
                           dataloader=datasets, 
                           verbose=1)


##############################
# 結果出力
##############################
plot.plotLoss(losses, start=1, end=hypParams['iteration'])

plot.plotAcc(losses, start=1, end=hypParams['iteration'])

