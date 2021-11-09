# ========================================
# import
# ========================================
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt

# ========================================
# モデルのクラス
# ========================================
class LinearRegression(nn.Module):
    # ----- コンストラクタ
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    # ----- ネットの定義
    def forward(self, x):
        y = self.linear(x)
        return y