#MLP用のモジュール作成

import torch
from torch import nn

class MLPBlock(nn.Module):
    '''
    MLPにおける各層の中身用のクラス
    '''
    def __init(self, )