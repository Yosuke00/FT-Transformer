#Feature_Tokenizer用のPythonファイルの作成
'''
[概要]
入力であるデータフレームを量的変数と質的変数の二つに分けて、それぞれ符号化する
最後に符号化した出力を結合して元のデータサイズと同じにする
'''

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
from transformers import AutoTokenizer

from activation_function import ReGLU, GEGLU


ModuleType = Union[str, Callable[..., nn.Module]]
def _is_glu_activation(activation:ModuleType):
    '''
    各GLU関数のインスタンス化
    '''
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )

def _all_or_none(values):
    '''
    データ内に欠損があるかどうかの判別を行う。(0か100の時にTrueを返す)
    '''
    return all(x is None for x in values) or all(x is not None for x in values)

class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    
    @classmethod #インスタンス化しなくてもクラスの内容を使用できる
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        '''
        cls:クラス
        '''
        try:
            return cls(initialization) #初期化用の確率分布の設定
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')
        
    def apply(self, x: Tensor, d:int) -> None: #出力なし
        d_sqrt_inv = 1 / math.sqrt(d) #正規化項
        if self == _TokenInitialization.UNIFORM: #一様分布による正規化
            #論文でも使われているパラメータの初期化設定
            #一様分布を使用
            nn.init.uniform_(x, a = -d_sqrt_inv, b = d_sqrt_inv)
            
        elif self == _TokenInitialization.NORMAL: #正規分布による正規化
            nn.init.normal_(x, std = d_sqrt_inv)
            
class NumericalFeatureTokenizer(nn.Module):
    '''
    量的変数をトークン化するクラス
    <内容>
    * 入力である量的変数に対して訓練可能なベクトル(パラメータw)をかける
    * もう一つの訓練可能ベクトル(バイアスb)を加える
    各訓練ベクトルは特徴量間で共有せずに別々の値を取る
    '''
    def __init__(self, 
                 n_features: int,
                 d_token : int,
                 bias : bool,
                 initialization: str,
                 ) -> None:
        '''
        [コンストラクタ]
        <param>
        n_features: 量的変数の数
        d_token: 一つ当たりのトークンサイズ
        bias : 'False':バイアス項ぬきでパラメータのみでトークン化する
               'True':通常通りバイアス項を加える
        initialization: パラメータの初期化手法の設定 ['uniform', 'Normal']どちらかを取る
        
        [参照]
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        '''
        #nn.Moduleの継承
        super().__init__()
        #パラメータの初期化手法の設定
        initialization_ = _TokenInitialization.from_str(initialization)
        #パラメータとバイアスの初期値の設定
        self.weight = nn.Parameter(Tensor(n_features, d_token))
        self.bias = nn.Parameter(Tensor(n_features, d_token)) if bias else None
        
        #初期化作業
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)
                
    @property #attributeとして抜き出せるように
    def n_tokens(self) -> int:
        '''
        トークン数の表示
        '''
        return len(self.weight)
    
    @property
    def d_token(self) -> int:
        '''
        トークンサイズの表示
        '''
        return self.weight.shape[1]
    
    def forward(self, x:Tensor) -> Tensor:
        '''
        学習用関数
        '''
        x = self.weight[None]*x[..., None] #Tensorサイズの調整
        if self.bias is not None:
            #バイアス項が存在している時xに加える
            x = x + self.bias[None]
        return x #1×n_num_features×d_token

class CategoricalFeatureTokenizer(nn.Module):
    '''
    質的変数のトークン化用のクラス
    torchにもともとあるnn.Embeddingを利用
    '''
    category_offsets: Tensor
    
    def __init__(
        self, 
        cardinalities: List[int],
        d_token : int,
        bias : bool,
        initialization: str,
    ) -> None:
        '''
        [コンストラクト]
        <param>
        cardinalities: 各特徴量の離散値の数
        d_token: 一つ当たりのトークンサイズ
        bias : 'False':バイアス項ぬきでパラメータのみでトークン化する
               'True':特徴量の値に関係なく各特徴量ごとでバイアス項の値が決まる
        initialization: パラメータの初期化手法の設定 ['uniform', 'Normal']どちらかを取る
        '''
        #nn.Moduleの継承
        super().__init__()
        #条件を満たさないとエラーが出るよう指定
        assert cardinalities, 'cardinalities must be non-empty'#中身がないとエラーを吐く
        assert d_token > 0,'d_token must be positive'
        
        #パラメータの初期化手法の設定
        initialization_ = _TokenInitialization.from_str(initialization)
        
        #一意のベクトルで表すために累積和を利用
        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        
        #訓練パラメータから累積和をbufferに登録する
        self.register_buffer('category_offsets', category_offsets, persistent = False)
        
        #Embeddingとbiasの設定
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None
        
        #パラメータの更新
        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)
        
    @property
    def n_tokens(self) -> int:
        '''
        トークン数(カラム数)
        '''
        return len(self.category_offsets)
    
    @property
    def d_token(self) -> int:
        '''
        1トークンに含まれるユニーク数
        '''
        return self.embeddings.embedding_dim #各トークンのユニーク数の合計
    
    def forward(self, x:Tensor) -> Tensor:
        '''
        学習用関数
        x:質的変数をTensorにしたもの
        '''
        x = self.embeddings(x + self.category_offsets[None]) 
        if self.bias is not None:
            x = x + self.bias[None]
        return x #1×sum(cardinalities)×d_token
    
class FeatureTokenizer(nn.Module):
    '''
    NumericalFeatureTokenzer, CategoricalFeatureTokenizerを結合する
    このモデルは量的変数と質的変数をトークンに変換する
    [参照]
    * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
    '''
    def __init__(
        self, 
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        '''
        [コンストラクト]
        <param>
        n_num_features: 量的変数のカラム数
                        0の時は量的変数0を表す
        cat_cardinalities: 各質的変数カラムのユニーク数
        d_token: 1トークン当たりのサイズ
        '''
        #パラメータの継承
        super().__init__()
        
        #警告の設定
        #量的変数が0以下でエラーを吐く
        assert n_num_features >= 0, 'n_num_features must be non_negative'
        #特徴量が一つも入っていないとエラーを吐く
        assert(
            n_num_features or cat_cardinalities
        ), 'at least one of n_num_features or cat_cardinalities must be positive/non_empty'
        #パラメータの初期化(一様分布で固定)
        self.initalization = 'uniform'
        
        #量的変数用のトークナイザ
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features = n_num_features,
                d_token = d_token,
                bias = True,
                initialization = self.initalization
            )
            #量的変数の有無で条件分岐
            if n_num_features
            else None
        )
        #質的変数用のトークナイザ
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initalization
            )
            #質的変数の有無で条件分岐
            if cat_cardinalities
            else None
        )
        self.d = d_token
    @property    
    def n_tokens(self) -> int:
        '''
        トークン数(特徴量のカラム数)
        '''
        return sum(
            x.n_tokens
            for x in [self.num_tokenizer, self.cat_tokenizer] #各種tokenizerの出力を一つのリストにまとめる
            if x is not None
        )
    @property
    def d_token(self) -> int: #
        '''
        1トークン当たりのサイズ
        '''
        return (
            self.cat_tokenizer.d_token
            if self.num_tokenizer is None
            else self.num_tokenizer.d_token #d_tokenを指定したため
        )
    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor], text_vect: Tensor) -> Tensor: 
        #Optionalを用いて型を明示することで明示した型に加えてNoneが入っても大丈夫なようにしている
        '''
        学習用の関数
        [コンストラクタ]
        <param>
        x_num: 特徴量のうち量的変数なもの
               n_num_features >0である必要があるときに明示する必要がある
        x_cat: 特徴量のうち質的変数なもの
               cat_cardinalitiesがnot emptyである必要があるときに明示する必要がある
        <output>
        tokens: 量的変数、質的変数両方のトークンを返す
        <AssertionError>
        各関数のAssertionErrorに該当したとき
        '''
        #各種AsserttionErrorの指定
        assert(
            x_num is not None or x_cat is not None
        ), 'At least one of x_num and x_cat must be presented'
        assert(
            [self.num_tokenizer, x_num]
        ), 'If self.num_tokenizer is (not) None, then x_num must (not) be None'
        assert(
            [self.cat_tokenizer, x_cat]
        ), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        
        #xにそれぞれトークン化したものを入れる
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        
        if text_vect is not None:
            x.append(text_vect)
        return x[0] if len(x) == 1 else torch.cat(x, dim = 1) #1×(n_num_features + sum(cardinalities))×d_token
    
class CLSToken(nn.Module):
    '''
    BERTのようにTokenのはじめに特殊トークン[CLS]を組み込む
    [CLS]トークンはバッチの句切れとして用いる
    [参照]
    * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    '''
    def __init__(self, d_token: int, initialization: str) -> None:
        '''
        [コンストラクト]
        <param>
        d_token:トークンのサイズ
        intialization: パラメータの初期化手法の設定([normal , uniform]いずれかを選択)
        [参照]
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        '''
        #パラメータの継承
        super().__init__()
        
        #初期化手法の設定
        initialization_ = _TokenInitialization.from_str(initialization)
        #重みの設定
        self.weight = nn.Parameter(Tensor(d_token))
        #重みの初期化
        initialization_.apply(self.weight, d_token)
        
    def expand(self, *leading_dimensions: int) -> Tensor:
        '''
        全ての列に[CLS]を加え他と同義にするために各行に1を加える
        Note:
        この関数は'torch.Tensor.expand'に基づいているため勾配は保証されている.
        [コンストラクト]
        <param>
        leading_dimensions: 追加する次元 (List型)
        <output>
        w:tensor of the shape(*leading_dimensions, len(self.weight))
        '''
        #次元の追加がない場合はそのまま重みを渡す
        if not leading_dimensions:
            return self.weight
        #それ以外の場合はevent_shapeに1次元追加する
        #要素が1の追加する分の次元数の設定
        new_dims = (1,) * (len(leading_dimensions) -1)
        #重みの拡張を行う
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)
        #viewで次元の配列を変換し、expandで同じ要素の配列を用いて拡張する.
        
    def forward(self, x: Tensor) -> Tensor:
        '''
        学習用の関数
        各バッチの最後に[CLS]トークンを追加する
        '''
        return torch.cat([x, self.expand(len(x), 1)], dim = 1)
        