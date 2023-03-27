#活性化関数用のPythonファイル

import enum
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor

class ReGLU(nn.Module):
    '''
    ReGLU活性化関数用のクラス
    '''
    def reglu(self, x:Tensor):
        '''
        参照:[1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
        '''
        assert x.shape[-1] % 2 == 0 #-1次元が2で割り切れることを保証している
        a, b = x.chunk(2, dim = -1) #テンソルの分割(一番後ろの次元を2グループに分割)
        return a*F.relu(b)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.reglu(x)
    
class GEGLU(nn.Module):
    '''
    GEGLU活性化関数用のクラス
    ''' 
    def gegle(self, x:Tensor) -> Tensor:
        '''
        参照:[1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
        '''
        assert x.shape[-1] % 2 == 0 #-1次元が2で割り切れることを保証している
        a, b = x.chunk(2, dim = -1) #テンソルの分割(一番後ろの次元を2グループに分割)
        return a * F.gelu(b)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.gegle(x)
    
        