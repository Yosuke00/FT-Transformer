#推論用の関数パッケージ
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from scipy.special import expit

import torch 
import torch.nn as nn
import torch.nn.functional as F

from activation_function import ReGLU, GEGLU
from Feature_Tokenizer import FeatureTokenizer, CLSToken
import model_
from model_ import FTTransformer

class Train:
    '''
    推論用のクラス作成
    '''
    def __init__(
        self, model: FTTransformer,
        x_num_features: int,
        cat_cardinalities: int,
        d_out: int,
        task_type: str,
        last_layer_query_idx: list = None,
        device: str = None,
        lr = 0.001,
        weight_decay = 0.0
        ):
        '''
        loss_fn:
        二値分類 -> F.binary_cross_entropy_with_logits
        多クラス分類 -> F.cross_entropy
        回帰 -> F.mse_loss
        '''
        #modelの定義
        if last_layer_query_idx:
            self.model = model.make_default(
                n_num_features = x_num_features,
                cat_cardinalities = cat_cardinalities,
                last_layer_query_idx = last_layer_query_idx,
                d_out = d_out
            )
        else:
            self.model = model.make_default(
                n_num_features = x_num_features,
                cat_cardinalities = cat_cardinalities,
                last_layer_query_idx = None,
                d_out = d_out
            )
        #GPUに載せるか否か
        if device:
            self.model.to(device)
        #最適化手法の設定
        self.optimizer = (
            self.model.make_default_optimizer()
            if isinstance(self.model, FTTransformer)
            else torch.optim.AdamW(self.model.parameters(), lr = lr, weight_decay= weight_decay)
        )
        #損失の設定
        self.loss_fn = (
            #二値交差エントロピー(Logitsでの損失を算出ver)
            F.binary_cross_entropy_with_logits
            if task_type == 'binclass'
            #交差エントロピー
            else F.cross_entropy
            if task_type == 'multiclass'
            #MSE
            else F.mse_loss
        )
        #タスクタイプの保存
        self.task_type = task_type
    @torch.no_grad()
    def evaluate(self, model: FTTransformer, data: torch.Tensor,
                 Y:torch.Tensor, len_num :int, labels:list[int] = None) -> np.array: 
        '''
        モデルの評価用関数
        [コンストラクト]
        <param>
        model: FT-Transformer
        data_loader:評価対象のデータ
        Y:データにおける目的変数
        len_num: 量的変数のカラム数
        labels: 目的変数のユニーク値
        '''
        #評価モードに切り替え
        model.eval()
        #予測値の算出
        prediction = []
        for _, batch in enumerate(data):
            #量的変数と質的変数に分割
            x_num_batch = batch[:len_num]
            x_cat_batch = batch[len_num:-1]
            if len_num == 1:
                x_num_batch = x_num_batch.reshape(-1, 1)
            prediction.append(model(x_num_batch, x_cat_batch))
        #予測値の次元数を整える
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        #目的変数の準備
        target = Y.cpu().numpy()
        #タスクによって変更
        if self.task_type == 'binclass': #二クラス分類
            prediction = np.round(expit(prediction))
            score = accuracy_score(target, prediction) #正解率
        elif self.task_type == 'multiclass': #多クラス分類
            score = log_loss(target, prediction, labels = labels) #対数損失
        else :
            assert self.task_type == 'regression'#回帰
            #MSE
            score = mean_squared_error(target, prediction) ** 0.5 * Y.std().item()
        
        return score
    
    def train(self, n_epochs:int, len_num:int,
              train_loader: torch.utils.data.DataLoader,
              val_data: torch.Tensor,
              y_val: torch.Tensor,
              labels = None
              ):
        '''
        学習用関数
        [コンストラクト]
        <param>
        n_epochs:エポック数
        report_freq:表示頻度
        data_loader:学習させるデータ
        '''
        model = self.model
        max_val_score = 0
        for epoch in range(1, n_epochs + 1):
            for iter, batch in enumerate(train_loader):
                model.train()
                self.optimizer.zero_grad()
                x_num_batch = batch[:, :len_num]
                x_cat_batch = batch[:, len_num:-1]
                y_batch = batch[:, -1]
                y_batch = y_batch.type(torch.float).reshape(-1, 1)
                if len_num == 1:
                    x_num_batch = x_num_batch.reshape(-1, 1)
                loss = self.loss_fn(model(x_num_batch, x_cat_batch).squeeze(1), y_batch)
                loss.backward()
                self.optimizer.step()
            val_score = self.evaluate(model, val_data, y_val, len_num = len_num, labels = labels)
            print(f'Epoch:{epoch:03d} | val_score:{val_score:.4f}')
            if epoch == 1:
                max_val_score = val_score
            elif max_val_score < val_score:
                max_val_score = val_score
                torch.save(model.state_dict(), f"./model.pth")
    