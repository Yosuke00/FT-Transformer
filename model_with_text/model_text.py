#FT-Transformer用のTransformer
#線形層がAttention層の前にあるのが特徴(論文曰く精度が上がるらしい)

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

from activation_function import ReGLU, GEGLU
from FT_text import FeatureTokenizer, CLSToken

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'

def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)

def _make_nn_module(module_type: ModuleType,  *args) -> nn.Module:
    '''
    活性化関数の設定用関数
    '''
    if isinstance(module_type, str): #module_typeがstrの時はそいつをインスタンス化
        if module_type == 'ReGLU':
            return ReGLU()
    
        elif module_type == 'GEGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err #該当しない場合はエラーをはく
            return cls(*args) #torch.nnにある場合はそれをインスタンス化する
    else:
        return module_type(*args) #module_typeがstr出ない時もインスタンス化
            
class MultiHeadAttention(nn.Module):
    '''
    MultiHeadAttentionだがLinformerで言われている通り、線形変換を行ってからAttentionに入れる
    [参照]
    * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    * [wang2020linformer] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    '''
    def __init__(
        self, 
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        '''
        [コンストラクト]
        <param>
        d_token:トークンサイズ
        n_heads: Attentionヘッド数
        dropout: ドロップアウトの割合
        bias: バイアス項の有無
        initialization: 初期化方法の設定
        <AssertionError>
        入力要件を満たしていないときにエラーが出る
        '''
        #パラメータの継承
        super().__init__()
        
        #assertの定義
        #d_tokensはヘッド数の倍数でないといけない
        if n_heads > 1:
            assert d_token % n_heads == 0, 'd_token must be multiple of n_heads'
        #初期化手法の制限(kaimingが提案した手法 or xavierが提案した手法)
        assert initialization in ['kaiming', 'xavier']
        
        #K, Q, Vの線形変換(Linformer)用の重みwの設定()
        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        #Attentionヘッドの設定
        self.n_heads = n_heads
        #ドロップアウトの設定
        self.dropout = nn.Dropout(dropout) if dropout else None
        #各重みの初期化
        for m in [self.W_q, self.W_k, self.W_v]:
            #'xavier'による手法が一般的であるが条件分岐次第で変更する
            #条件2はW_vがW_outの役割を担うか確認している
            #各条件を満たさないときは'Kaiming'による手法で初期化をおこなっている。
            if initialization == 'xavier' and (
                m is not self.W_v or self.W_out is not None
            ):
                #xavier版の一様分布　による初期化
                nn.init.xavier_uniform_(m.Weight, gain = 1 / math.sqrt(2))
            #バイアス項の有無で条件分岐
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            
        #MultiHeadかどうかで条件分岐
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)
            
    
    def _reshape(self, x: Tensor) -> Tensor:
        '''
        入力のshapeの変換
        ex)ヘッド数 8
        x.shape = (4, 4, 16)
        ↓ 変換後
        x.shape = (32, 4, 2)
        '''
        #各種サイズの設定
        batch_size, n_tokens, d = x.shape
        #ヘッド数の設定
        d_head = d // self.n_heads #割って切り捨て
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )
        
    def forward(
        self, 
        x_q: Tensor, 
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''
        推論用の関数
        [コンストラクト]
        <param>
        x_q: クエリートークン
        x_kv: キー、バリュートークン
        key_compression: Linformer用のkeyのための線形変換層
        value_compression: Linformer用のvalueのための線形変換層
        <output>
         (token, attention_state)
        '''
        #どちらかが欠けていたらエラーを吐く
        assert _all_or_none(
            [key_compression, value_compression]
        ), 'If key_compression is (not) None, then value_compression must (not) be None'
        
        #query, key, valueの設定
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k ,v]:
        #tensorがヘッド数で割り切れない時エラー
            assert tensor.shape[-1] % self.n_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2) #type ignore(型はこだわらない)
            
        #各種サイズの設定
        batch_size = len(q) #バッチサイズ
        d_head_key = k.shape[-1] // self.n_heads #各ヘッドに入るkeyのサイズ
        d_head_value = v.shape[-1] // self.n_heads #各ヘッドに入るqueryのサイズ
        n_q_tokens = q.shape[1] #queryのトークン数
        
        #query, valueの準備
        q = self._reshape(q)
        k = self._reshape(k)
        
        #ロジット関数とSoftmaxの計算
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim = -1)
        #Dropoutの有無で条件分岐
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        #Attentionの出力の作成
        x = attention_probs @ self._reshape(v)
        #変形
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads*d_head_value)
        )
        #出力の重みの有無で条件分岐
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            'attention_logits': attention_logits,
            'attention_probs' : attention_probs,
        }
        
class Transformer(nn.Module):
    '''
    FT-Transformerの元となるTransformer
    正規化層の場所が異なる。
    '''
    WARNINGS = {'fisrt_prenormalization': True, 'prenormlization': True}
    class FFN(nn.Module):
        '''
        Fees-Forwardネットワーク用のクラス
        '''
        def __init__(
            self, 
            d_token: int,
            d_hidden : int,
            bias_first: bool, 
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            #パラメータの継承
            super().__init__()
            #最初の線形層の設定
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1), 
                #GLU関数は入力を二つに分割して、その後アダマール積をとる
                bias_first
            ) # 入力をd_tokenからd_hidden次元に変換
            #活性化関数の設定
            self.activation = _make_nn_module(activation)
            #dropoutの設定
            self.dropout = nn.Dropout(dropout)
            #二層目の線形層の線形層の設定
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)
            #入力をd_hiddenからd_tokenに変換
            
        def forward(self, x: Tensor) -> Tensor:
            '''
            推論用の関数
            '''
            #線形層
            x = self.linear_first(x)
            #活性化関数
            x = self.activation(x)
            #Dropout
            x = self.dropout(x)
            #2個目の線形層
            x = self.linear_second(x)
            return x
    
    class Average(nn.Module):
        '''
        推論に用いる時に用いる(BERTみたいな感じ)
        '''
        def __init__(
            self,
            *,
            d_in: int,
            bias :bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            '''
            [コンストラクト]
            <param>
            d_in: 入力の次元
            bias: バイアス項の有無
            activation: 活性化関数の設定
            normalization: 標準化の設定
            d_out: 出力の次元
            '''
            #パラメータの継承
            super().__init__()
            
            #各層の設定
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)
            
        def forward(self, x: Tensor) -> Tensor:
            '''
            推論用の関数
            '''
            #平均の取得
            x = torch.mean(x, dim = 1, keepdim = True)
            #正規化
            x = self.normalization(x)
            #活性化関数の通過
            x = self.activation(x)
            #線形変換
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_token: int, #トークンの次元数
        n_blocks: int , #Attention部分の層数
        attention_n_heads: int, #AttentionHead数
        attention_dropout: float, #AttentionにおけるDropoutの割合指定
        attention_initialization: str, #Attentionのパラメータの初期化手法の設定
        attention_normalization: str, #Attentionの正規化手法の設定
        ffn_d_hidden: int, #FFNの隠れ層の次元数
        ffn_dropout: float, #FFNのDropoutの割合指定
        ffn_activation: str, #FFNの活性化関数の指定
        ffn_normalization: str, #FFNの正規化手法の設定
        residual_dropout: float, #残差接続部分のDropout割合の指定
        prenormalization: bool, #このTransformerの特徴である事前正規化について
        first_prenormalization: bool, #Attentionの前の正規化をどうするかについて
        last_layer_query_idx: Union[None, list[int], slice], #最終層のqueryのインデックス
        n_tokens: Optional[int], #トークン数
        kv_compression_ratio: Optional[float], #keyとvalueの割合を決める
        kv_compression_sharing: Optional[str], #情報の共有方法について
        head_activation: ModuleType, #推論時に使う値のための活性化関数
        head_normalization: ModuleType, #推論時に使う値のための正規化手法
        d_out: int, #出力の次元数
    ) -> None:
        #パラメータの継承
        super().__init__()
        #インデックス部分がint型以外を取る場合はエラーを吐く
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                'last_layer_query_idx must be None, List[int] or slice.'
                f'Do you mean last_layer_query_idx = [{last_layer_query_idx}] ?'
            )            
        #prenormalizationがないかつfirst_prenormalizationがない時エラーを吐く
        if not prenormalization:
            assert(
                not first_prenormalization
            ), 'If "prenormalization" is False, then "first_prenormalization" must be False'
        #n_tokens, kv_compression_ratio, kv_compression_sharingのいずれかが欠けているとエラーを吐く
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]),(
            'If any of the following arguments is (not) None, then all of them must (not) be None:'
            'n_tokens, kv_compression_ratio, kv_compression_sharing'
        )
        #kv_compression_sharingが該当しなければエラーを吐く
        assert kv_compression_sharing in [None, 'headwise', 'key_value', 'layerwise']
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn(
                    'prenormalization is set to False. Are you sure about this?'
                    'The training can become less stable.'
                    'You can turn off this warning by tweaking the '
                    'Transformer.WARNINGS dictionary.',
                    UserWarning,
                )
            assert (
                not first_prenormalization
            ), 'If prenormalization is False, then first_prenormalization is ignored and must be set to False'
        #prenormalizationとfirst_prenormalizationの両方を設定したらwarningを吐く
        if (
            prenormalization
            and first_prenormalization
            and self.WARNINGS['first_prenormalization']
        ):
            warnings.warn(
                'first_prenormalization is set to True. Are you sure about this?'
                'For example, the vanilla FT-Transformer with'
                'first_prenormalization= True performs SIGNIFICANLY worse.'
                'You can turn off this warning by tweaking the '
                ' Transformer.WARNINGS dictionary',
                UserWarning,
            )
            time.sleep(3)
            
        def make_kv_compression():
            '''
            Linformerにおけるkey, valueを線形変換する関数
            '''
            #n_tokensとkv_compression_ratioがNoneの時エラーを吐く
            assert (
                n_tokens and kv_compression_ratio
            ), _INTERNAL_ERROR_MESSAGE
            return nn.Linear(n_tokens, int(n_tokens*kv_compression_ratio), bias = False)
        
        #層間でkv_compressionを継承する場合のみ設定
        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == 'layerwise'
            else None
        )
        #正規化の設定
        self.prenormalization = prenormalization
        #最終層のインデックスの指定
        self.last_layer_query_idx = last_layer_query_idx
        #Transformerのブロック部分の設定
        self.blocks = nn.ModuleList([])
        #層のindexごとで考える
        for layer_idx in range(n_blocks):
            #層で取り出せるように辞書に格納
            layer = nn.ModuleDict(
                {
                    'attention': MultiHeadAttention(
                        d_token = d_token,
                        n_heads = attention_n_heads,
                        dropout = attention_dropout,
                        bias = True,
                        initialization= attention_initialization,
                    ),
                    'ffn':Transformer.FFN(
                        d_token = d_token,
                        d_hidden = ffn_d_hidden,
                        bias_first = True, 
                        bias_second = True,
                        dropout = ffn_dropout,
                        activation = ffn_activation
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'output': nn.Identity(), #逆伝播のため、、、?
                }
            )
            #Attention部分にだけ正規化を入れる時
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(
                    attention_normalization, d_token
                )
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            #kv_compression_ratioを指定していて、層間で共有しない場合
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                #先頭から順番に行く場合
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                #その他の場合かつ条件を満たさない時エラーを吐く
                else:
                    assert(
                        kv_compression_sharing == 'key-value'
                    ), _INTERNAL_ERROR_MESSAGE
            #できた層をself.blockに入れていく
            self.blocks.append(layer)
        #推論時に用いる関数の設定
        self.average = Transformer.Average(
            d_in = d_token,
            d_out = d_out,
            bias = True,
            activation=head_activation,
            normalization=head_normalization if prenormalization else 'Identity'
        )
    def _get_kv_compressions(self, layer):
        '''
        kv_compressionを定義するための関数
        '''
        return(
            (self.shared_kv_compression, self.shared_kv_compression)
        #self.shared_kv_compression の有無で条件分岐
        if self.shared_kv_compression is not None
        else (layer['key_compression'], layer['value_compression'])
        #さらに条件分岐,各compressionがともにlayerにあるか否かで分岐
        if 'key_compression' in layer and 'value_compression' in layer
        else (layer['key_compression'], layer['key_compression'])
        #key_compressionがlayerにあるかいないかでさらに条件分岐
        if 'key_compression' in layer
        else (None, None)
        )
    def _start_residual(self, layer, stage, x):
        '''
        残差接続のスタート地点を決定する関数
        '''
        #stageが'attention'と'ffn'どちらでもない時エラーを吐く
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        #prenormalizationgがある場合
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            #norm_keyがlayerに存在する場合
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual
    
    def _end_residual(self, layer, stage, x, x_residual):
        '''
        最終残差接続に関する関数
        '''
        #stageが'attention'と'ffn'どちらでもない時エラーを吐く
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        #残差接続するものの設定
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x+ x_residual
        #最終出力も正規化する
        x = layer[f'{stage}_normalization'](x)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        推論用の関数
        '''
        #入力の次元が(a, b, c)の三つないとエラーを吐く
        assert(
            x.ndim == 3
        ), 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        #各層に分割
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleList, layer) #辞書型からList型に変更
            #最終層に到着したらqueryのインデックスを保存
            query_idx = (
                self.last_layer_query_idx if layer_idx +1 == len(self.blocks) else None
            )
            #残差接続する値の取得
            x_residual = self._start_residual(layer, 'attention', x)
            #Attentionにかける
            x_residual, _ = layer['attention'](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            #最終層の時xの取得範囲を限定する
            if query_idx is not None:
                x = x[:, query_idx]
            #残差接続を行う
            x = self._end_residual(layer, 'ffn', x, x_residual)
            #恒等関数にいれる
            x = layer['output'](x)
        #推論に使用できる形にする
        x = self.average(x)
        return x
    
class FTTransformer(nn.Module):
    '''
    FT-Transformerの本体クラス
    "Feature-Tokenizer"を用いて特徴量をトークン化し、Transformerで推論を行う
    参照:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        * [vaswani2017attention] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", 2017
    '''
    def __init__(
        self, 
        feature_tokenizer: FeatureTokenizer,
        transformer: Transformer
    ) -> None:
        #パラメータの継承
        super().__init__()
        #Transformerにprenormalizatioinが指定されている時
        if transformer.prenormalization:
            #最初の層にprenormalizationがある時にエラーを吐く
            assert 'attention_normalization' not in transformer.blocks[0],(
                'In the prenormalization setting, FT-Transformer does not '
                'allow using the first normalization layer '
                'in the first transformer block'
            )
        #トークナイザとTransformerの設定
        self.feature_tokenizer = feature_tokenizer
        self.cls_token = CLSToken(
            feature_tokenizer.d_token, feature_tokenizer.initalization
        )
        self.transformer = transformer
        
    @classmethod
    def get_baseline_transformer_subconfig(
        cls: Type['FTTransformer'],
    ) -> Dict[str, Any]:
        '''
        Transformerのパラメータのベースラインを取得する関数
        '''
        return {
            'attention_n_heads':8, #ヘッド数
            'attention_initialization': 'kaiming',# 初期化手法
            'ffn_activation': 'ReGLU',#活性化関数
            'attention_normalization': 'LayerNorm', #Attentionのパラメータの正規化について
            'ffn_normalization':'LayerNorm', #FFNの正規化について
            'prenormalization': True, #正規化の順序について
            'first_prenormalization': False, #prenormalizationがTrueなため
            'last_layer_query_idx': None, #最終層のインデックスについて
            'n_tokens': None, #トークン数の指定
            'kv_compression_ratio': None, #LinformerにおけるK,Vの比重指定
            'kv_compression_sharing': None, #K, Vについて共有手法の選択
            'head_activation': 'ReLU', #Head部分の活性化関数について
            'head_normalization':'LayerNorm', #Head部分の正規化手法
        }
    @classmethod
    def get_default_transformer_config(
        cls: Type['FTTransformer'], *, n_blocks: int = 3
    ) -> Dict[str, Any]:
        '''
        各種パラメータのデフォルト値の設定
        n_blocksについては適宜変更する必要がある
        '''
        #n_blocksがTransformerの原論文の層数範囲を超えたらエラーを吐く 
        assert 1 <= n_blocks <= 6
        grid = {
            'd_token': [768, 768, 768, 768, 768, 768], #層の増加に伴うトークンの各次元
            'attention_dropout':[0.1, 0.15, 0.2, 0.25, 0.3, 0.35], #同様にAttention層におけるDropoutの割合指定
            'ffn_dropout':[0., 0.05, 0.1, 0.15, 0.2, 0.25],
        }
        #結果ではなくアーキないのパラメータを層数に合わせて設定
        arch_subconfig = {k: v[n_blocks - 1] for k, v in grid.items()}
        #モデルに組み込むのパラメータの設定
        baseline_subconfig = cls.get_baseline_transformer_subconfig()
        #GEGLU, ReGLUのときは3/4、ReLU, GELUのときは2を指定する。
        ffn_d_hidden_factor = (
            (3/4) if _is_glu_activation(baseline_subconfig['ffn_activation']) else 2.0
        )
        #各種設定したパラメータを出力として返す
        return {
            'n_blocks': n_blocks, #層数
            'residual_dropout': 0.0, #残差接続前のDropout
            'ffn_d_hidden': int(arch_subconfig['d_token'] * ffn_d_hidden_factor),  #隠れ層の出力の次元数
            **arch_subconfig, #アーキに組み込むパラメータ
            **baseline_subconfig #モデルに組み込むパラメータ
        }
    @classmethod
    def _make(
        cls, 
        n_num_features,
        cat_cardinalities,
        transformer_config,
    ):
        '''
        トークナイザとTransforomerのパラメータ指定したものを含むFT-Transformerを返す用の関数
        '''
        #トークナイザの指定
        feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features, #量的変数を含むカラム数
            cat_cardinalities=cat_cardinalities, #各質的変数カラムのユニーク数
            d_token = transformer_config['d_token'] #出力の次元数
        )
        #transformer内におけるパラメータに出力に関する次元の設定の有無で条件分岐
        if transformer_config['d_out'] is None:
            transformer_config['head_activatoin'] = None
        #kvの比重に関するパラメータの有無で条件分岐
        if transformer_config['kv_compression_ratio'] is not None:
            transformer_config['n_tokens'] = feature_tokenizer.n_tokens +1
        return FTTransformer(
            feature_tokenizer,
            Transformer(**transformer_config)
        )
    @classmethod
    def make_baseline(
        cls: Type['FTTransformer'], #クラスの指定
        *,
        n_num_features: int, #量的変数のカラム数
        cat_cardinalities: Optional[list[int]], #各質的変数カラムのユニーク数のリスト
        d_token: int, #トークンの次元数
        n_blocks: int, #層数
        attention_dropout: float, #attention層におけるDropoutの割合
        ffn_d_hidden: int, #ffnの隠れ層の次元数
        ffn_dropout: float, #ffnのdropoutの割合
        residual_dropout: float, #接続層におけるdropoutの割合
        last_layer_query_idx: Union[None, List[int], slice] = None, #最終層におけるqueryのインデックスに関するパラ
        kv_compression_ratio: Optional[float] = None,#keyとvalueの重みづけ
        kv_compression_sharing: Optional[str] = None,#keyとvalueの共有について
        d_out: int, #出力の次元数
    ) -> 'FTTransformer':
        '''
        これまで作成した関数を用いてFT-Transformerのベースラインを作成する
        パラメータの学習するための初期値を取得するのに用いる
        [コンストラクト]
        <param>
        それぞれのコメントアウトに該当部分書いているため注釈があるパラメータのみ記述
        attention_dropout: 値が大きいほどうまくいくらしい、、ほんと?
        last_layer_query_idx: 最終層の出力のうち[CLS]を除いたもののインデックス
        kv_compression_ratio:特徴量が膨大な時Attention機構の速度を上げるのに用いられる
                             ただ、予期せぬ答えが返ってくる可能性もあるため注意が必要
        [参照]
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
            * [wang2020linformer] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020       
        '''
        #各種初期パラメータの取得
        transformer_config = cls.get_baseline_transformer_subconfig()
        #それぞれの情報を関数内のものに変更する
        for arg_name in [
            'n_blocks', 
            'd_token',
            'attention_dropout',
            'ffn_d_hidden',
            'ffn_dropout',
            'residual_dropout',
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out'
        ]:
            #パラメータの更新
            transformer_config[arg_name] = locals()[arg_name]
            
        #更新したFT-Transformerを返す
        return cls._make(n_num_features, cat_cardinalities, transformer_config)
    @classmethod
    def make_default(
        cls:Type['FTTransformer'], #クラスの指定
        *,
        n_num_features: int, #量的変数のカラム数
        cat_cardinalities: Optional[List[int]], #質的変数の各カラムにおけるユニーク数
        n_blocks: int = 3, #層数
        last_layer_query_idx: Union[None, List[int], slice] = None, #最終層のインデックス
        kv_compression_ratio: Optional[float] = None, #key, valueの重みづけ
        kv_compression_sharing: Optional[str] = None, #共有手法について
        d_out: int, #出力の次元数
    ) -> 'FTTransformer':
        '''
        FT-Transformerの初期状態を作る関数
        "n_blocks = 3"については原論文をもとに設定
        * については最適化用の関数によって作成される構成要素
        複数FT-Transfoirmerがあるときはdefaultのアンサンブルを取るのが良い
        しかし、一種類のTransformerを用いるときはハイパラ調整が良い精度を出す
        '''
        #各種パラメータんの設定
        transformer_config = cls.get_default_transformer_config(n_blocks=n_blocks)
        #最新値に更新するパラメータの選択
        for arg_name in [
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out'
        ]:
            #段階的にパラメータを更新する
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(n_num_features, cat_cardinalities, transformer_config)
    
    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        '''
        パラメータの最適化する時に新しいパラメータを更新する用の関数
        '''
        #エラー要因の設定
        no_wd_names = ['feature_tokenizer', 'normalization', '.bias']
        
        #FTTransformer内に'feature_tokenizer'のアトリビュートがないとエラーを吐く
        assert isinstance(
            getattr(self, no_wd_names[0], None), FeatureTokenizer
        ), _INTERNAL_ERROR_MESSAGE
        
        #normalizationが関数内に存在して合計が指定したものと等しくないとエラーを吐く
        assert (
            sum(1 for name, _ in self.named_modules() if no_wd_names[1] in name)
            == len(self.transformer.blocks) * 2
            - int('attention_normalization' not in self.transformer.blocks[0]) #型タイプは無視
            + 1
        ), _INTERNAL_ERROR_MESSAGE
        
        def needs_wd(name):
            '''
            "no_wd_names"のうちnameに含まれないものを返す関数
            '''
            return all(x not in name for x in no_wd_names)
        
        #重みを置くパラメータと重みを消すパラメータそれぞれが入った2種類の辞書を返す
        return [
            {'params': [v for k, v in self.named_parameters() if needs_wd(k)]},
            {
                'params': [v for k, v in self.named_parameters() if not needs_wd(k)],
                'weight_decay': 0.0,
            },
        ]
    
    def make_default_optimizer(self) -> torch.optim.AdamW:
        '''
        FT-Transformerのデフォルトのパラメータ最適化手法を作成する手法
        '''
        return torch.optim.AdamW(
            self.optimization_param_groups(),
            lr = 1e-4,
            weight_decay = 1e-5,
        )
    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor], text_vect: Optional[Tensor]) -> Tensor:
        '''
        推論用関数
        '''
        x = self.feature_tokenizer(x_num, x_cat, text_vect)
        x = self.cls_token(x)
        x = self.transformer(x)
        return x
        
        
        