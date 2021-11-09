##############################
# import
##############################
# 学習、テスト関数
from utils.train import TrainClass
# モデル
from models.model import LinearRegression
# 学習、テストデータ
from utils.datasets import loadRandData

##############################
# パラメータ設定
##############################
trainClass = TrainClass()
trainClass.fixSeed() # 乱数の固定
hypParams = {
    'in_features': 2,
    'out_features': 1,
    'lr': 0.01,
    'iteration': 500
}

# モデルの設定
model = LinearRegression(in_features=hypParams['in_features'], 
                         out_features=hypParams['out_features'])
# 最適化
optimizer = optim.SGD(model.Parameters(), lr=hypParams['lr'])
# 損失関数
E = nn.MSELoss()

##############################
# データ取得
##############################
X_train, y_train, X_test = loadRandData()


##############################
# 学習
##############################
model, losses = trainClass._train(model=model, 
                       optimizer=optimizer, 
                       E=E, 
                       iteration=hypParams['iteration'], 
                       x=X_train, 
                       y=y_train, 
                       verbose=1)

##############################
# テスト
##############################
testLoss = trainClass.test()
