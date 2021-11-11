# ========================================
# import
# ========================================
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as f

# ========================================
# モデルのクラス
# ========================================
class MyNet(nn.Module):
    # ----- コンストラクタ
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1000)
        self.fc2 = nn.Linear(1000, out_features)
    # ----- ネットの定義
    def forward(self, x):
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y)
        
        y = f.log_softmax(y, dim=1)
        
        return y