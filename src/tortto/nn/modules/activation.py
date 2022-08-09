import tortto as tt
from .linear import Linear
from tortto.nn.init import constant_, xavier_normal_, xavier_uniform_
from tortto.nn.parameter import Parameter
from .module import Module
from .. import functional as F


class Tanh(Module):
    def forward(self, tensor):
        return tt.tanh(tensor)


class Sigmoid(Module):
    def forward(self, tensor):
        return tt.sigmoid(tensor)


class LogSigmoid(Module):
    def forward(self, tensor):
        return F.logsigmoid(tensor)


class ReLU(Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        return F.relu(tensor, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class LeakyReLU(Module):
    def __init__(self, negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, tensor):
        return F.leaky_relu(tensor, self.negative_slope, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return f'negative_slope={self.negative_slope}{inplace_str}'

class GELU(Module):
    def forward(self, tensor):
        return F.gelu(tensor)

class Softmax(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return F.softmax(tensor, self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class LogSoftmax(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return F.log_softmax(tensor, self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'

class MultiheadAttention(Module):
    __constants__ = ['batch_first']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(tt.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(tt.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(tt.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(tt.empty((3 * embed_dim, embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(tt.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(tt.empty((1, 1, embed_dim)))
            self.bias_v = Parameter(tt.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True):

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.swapaxes(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)

        if self.batch_first and is_batched:
            return attn_output.swapaxes(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
