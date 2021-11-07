# ========================================
# 学習の際に用いるクラス
# ========================================



# ========================================
# import
# ========================================
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt



# ========================================
# TrainClass
# ========================================


class TrainClass():

    
    def train(self, model, optimizer, E, iteration, dataloader, valid=False, verbose=1):
        """
        モデルの学習時に呼び出し

        Parameters
        ----------
        model : torch.nn.Module
            学習対象のモデル
        optimizer : torch.optim
            最適化手法
        E : function
            損失関数
        iteration : int
            学習回数
        dataloader : dict
            入力値, 正解ラベル
        valid : bool
            validを実施するか
        verbose : int
            0=詳細を非表示
            1=詳細を表示

        Returns
        -------
        model : torch.nn.Module
            学習済みのモデル
        losses : dict
            各ループごとの損失リスト(train, valid)
        """

        losses = {
            'train_loss': [],
            'valid_loss': []
        }

        # 学習用ループ
        for i in range(iteration):

            # ----- train -----
            model.train(True)

            # dataloader['train']に含まれているデータ分ループ
            train_losses = []
            for i, (data, target) in enumerate(dataloader['train']):
                optimizer.zero_grad()   # 勾配初期化

                y_pred = model(data)       # 予測

                loss = E(y_pred, target)
                loss.backward()
                optimizer.step()        # 勾配更新

                train_losses.append(loss.item())  # 損失の保存
            
            # 1epoch分の損失(dataloader分の平均損失)を格納
            train_loss = sum(train_losses) / len(train_losses)
            losses['train_loss'].append(train_loss)




            # ----- valid -----
            # validがtrueであれば、validの損失を計算
            if valid:
                model.train(False)

                with torch.no_grad():
                    # dataloader['valid']に含まれているデータ分ループ
                    valid_losses = []

                    for i, (data, target) in enumerate(dataloader['valid']):
                        y_pred = model(data)       # 予測

                        loss = E(y_pred, target)

                        valid_losses.append(loss.item())  # 損失の保存

                    # 1epoch分の損失(dataloader分の平均損失)を格納
                    valid_loss = sum(valid_losses) / len(valid_losses)
                    losses['valid_loss'].append(valid_loss)



            # ----- verboseにより詳細の表示・非表示 -----
            if verbose != 0:
                outputText = 'epoch(%d) : loss(train)=%f' % (i+1, train_loss)
                if valid:
                    outputText += ' loss(valid)=%f' % (valid_loss)

                print(outputText)

        return model, losses


    def test(self, model, E, dataloader, verbose=1):
        """
        モデルのテスト時に呼び出し

        Parameters
        ----------
        model : torch.nn.Module
            テスト対象のモデル
        E : function
            損失関数
        dataloader : dict
            入力値, 正解ラベル
        verbose : int
            0=詳細を非表示
            1=詳細を表示

        Returns
        -------
        model : torch.nn.Module
            学習済みのモデル
        losses : dict
            各ループごとの損失リスト(train, valid)
        """

        # ----- test -----
        model.train(False)

        with torch.no_grad():
            # dataloader['test']に含まれているデータ分ループ
            test_losses = []

            for i, (data, target) in enumerate(dataloader['test']):
                y_pred = model(data)       # 予測

                loss = E(y_pred, target)

                test_losses.append(loss.item())  # 損失の保存

            # 1epoch分の損失(dataloader分の平均損失)を格納
            test_loss = sum(test_losses) / len(test_losses)



        # ----- verboseにより詳細の表示・非表示 -----
        if verbose != 0:    
            print('loss(test):%f' % (test_loss))

        return test_loss



    # =======================================
    def _train(self, model, optimizer, E, iteration, x, y, verbose=1):
        """
        モデルの学習時に呼び出し

        Parameters
        ----------
        model : torch.nn.Module
            学習対象のモデル
        optimizer : torch.optim
            最適化手法
        E : function
            損失関数
        iteration : int
            学習回数
        x : Tensor
            入力値
        y : Tensor
            正解ラベル
        verbose : int
            0=詳細を非表示
            1=詳細を表示

        Returns
        -------
        model : torch.nn.Module
            学習済みのモデル
        losses : list
            各ループごとの損失リスト
        """

        losses = []

        # 学習用ループ
        for i in range(iteration):
            optimizer.zero_grad()   # 勾配初期化

            y_pred = model(x)       # 予測

            loss = E(y_pred.reshape(y.shape), y)
            loss.backward()
            optimizer.step()        # 勾配更新

            losses.append(loss.item())  # 損失の保存

            # verboseにより詳細の表示・非表示
            if verbose != 0:
                print('epoch(%d) : loss=%f' % (i+1, loss.item()))

        return model, losses



