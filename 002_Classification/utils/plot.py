# ========================================
# import
# ========================================
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt

class Plot():
    # ========================================
    # グラフ描画
    # ======================================== 
    def plot(self, x, y, x_new, y_pred, losses):
        # フォントの種類とサイズを設定する。
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Times New Roman'
    
        # 目盛を内側にする。
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        
        
        # グラフの上下左右に目盛線を付ける。
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(121)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax2 = fig.add_subplot(122)
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
    
        # 軸のラベルを設定する。
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('E')
    
        # スケール設定
        ax1.set_xlim(-10, 20)
        ax1.set_ylim(0, 30)
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0.1, 100)
        ax2.set_yscale('log')
    
        # データプロット
        ax1.scatter(x, y, label='dataset')
        ax1.plot(x_new, y_pred, color='red', label='PyTorch result', marker="o")
        ax2.plot(np.arange(0, len(losses), 1), losses)
        ax2.text(600, 30, 'Loss=' + str(round(losses[len(losses)-1], 2)), fontsize=16)
        ax2.text(600, 50, 'Iteration=' + str(round(len(losses), 1)), fontsize=16)
    
        # グラフを表示する。
        ax1.legend()
        fig.tight_layout()
        plt.show()
        plt.close()

        
    # ========================================
    # Lossグラフ描画
    # ======================================== 
    def plotLoss(self, losses, start=1, end=100):
        plt.figure()
        plt.plot(range(start, end+1), losses['train_loss'], label='train_loss')
        plt.plot(range(start, end+1), losses['valid_loss'], label='valid_loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
        
    # ========================================
    # Accグラフ描画
    # ======================================== 
    def plotAcc(self, losses, start=1, end=100):
        plt.figure()
        plt.plot(range(start, end+1), losses['valid_acc'], label='valid_acc')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        plt.close()
        
    # ========================================
    # 出力画像描画
    # ======================================== 
    def plotOutput(self, images, height=3, width=2):
        for i in range(height * width):
            ax = plt.subplot(width, height, i+1)
            img = images[i].to('cpu').detach().numpy().transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis("off")
        plt.show()
        plt.close()

