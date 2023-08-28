from ..autograd.grad_nn import *

_floating_point = {float16, float32, float64}


def relu(input, inplace=False):
    return Relu.apply(input, inplace=inplace)


def relu_(input):
    return Relu.apply(input, inplace=True)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    return LeakyRelu.apply(input, negative_slope=negative_slope, inplace=inplace)


def leaky_relu_(input, negative_slope=0.01):
    return LeakyRelu.apply(input, negative_slope=negative_slope, inplace=True)


def gelu(input, approximate='none'):
    return Gelu.apply(input, approximate=approximate)


def mse_loss(input, target, reduction='mean'):
    return MseLoss.apply(input, target, reduction=reduction)


def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    return BinaryCrossEntropy.apply(input, target, weight=weight, reduction=reduction)


def binary_cross_entropy_with_logits(input, target, weight=None, pos_weight=None, reduction='mean'):
    return BinaryCrossEntropyWithLogits.apply(input, target, weight=weight, pos_weight=pos_weight, reduction=reduction)


def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    return NllLoss.apply(input, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)


def softmax(input, dim):
    return Softmax.apply(input, dim=dim)


def log_softmax(input, dim):
    return LogSoftmax.apply(input, dim=dim)


def logsigmoid(input):
    return LogSigmoid.apply(input)


def linear(input, weight, bias):
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    # linear is defined as y=X@A.T+b
    requires_grad = input.requires_grad | weight.requires_grad
    if bias is not None:
        output = matmul(input, weight.T) + bias
        requires_grad |= bias.requires_grad
    else:
        output = matmul(input, weight.T)
    return output


def pad(input, pad, mode='constant', value=0.0):
    '''
    TODO: implement cache
    TODO: only support padding with constant for now
    '''
    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    assert len(pad) // 2 <= input.dim(), "Padding length too large"

    if mode == 'constant':
        return ConstantPad.apply(input, pad=pad, value=value)
    else:
        raise NotImplementedError("TODO: Currently only support mode='constant'")


def conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    low_dim = False
    if input.ndim == 3:
        low_dim = True
        input = Unsqueeze.apply(input, dim=0)
    result = Convolution.apply(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if low_dim:
        result = Squeeze.apply(result, dim=0)
    return result


def conv_transpose2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1,
                     dilation=(1, 1)):
    low_dim = False
    if input.ndim == 3:
        low_dim = True
        input = Unsqueeze.apply(input, dim=0)
    result = TransposedConvolution.apply(input, weight, bias, stride=stride, padding=padding,
                                         output_padding=output_padding, dilation=dilation, groups=groups)
    if low_dim:
        result = Squeeze.apply(result, dim=0)
    return result


def max_pool2d(input, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), ceil_mode=False,
               return_indices=False):
    return MaxPool2DWithIndices.apply(input, kernel_size=kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)


def dropout(input, p=0.5, training=True, inplace=False):
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    if training and p > 0.0:
        return Droupout.apply(input, p=p, inplace=inplace)
    else:
        return input


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    return BatchNorm.apply(input, weight, bias, running_mean=running_mean, running_var=running_var,
                           training=training, momentum=momentum, eps=eps)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return LayerNorm.apply(input, weight, bias, normalized_shape=normalized_shape, eps=eps)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return Embedding.apply(weight, input=input, padding_idx=padding_idx, max_norm=max_norm,
                           norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)


def _in_projection_packed(q, k, v, w, b=None):
    """
    shape:
    - q: (tgt, bsz, E)
    - k: (src, bsz, E)
    - v: (src1, bsz, E) # may be different than k
    - w: (E * 3, E)
    - b: (E * 3)
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)

        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(q, k, v, w_q, w_k, w_v, b_q=None, b_k=None, b_v=None):
    """
    - q: (Qdims..., Eq), k: (Kdims..., Ek), v: (Vdims..., Ev)
    where Eq,Ek,Ev are the query,key,value embedding dimensions and Qdims,Kdims,Vdims are any
    number of leading dimensions.
    - w_q: (Eq, Eq), w_k: (Eq, Ek), w_v: (Eq, Ev)
    - b_q: (Eq), b_k: (Eq), b_v: (Eq)

    Output:
     - q':(Qdims..., Eq), k':(Kdims..., Eq), v':(Vdims..., Eq)
    """
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_dot_product_attention(q, k, v, attn_mask=None, training=False, dropout_p=0.0):
    bsz, tgt, E = q.shape
    q = q / math.sqrt(E)
    # (bsz, tgt, E) @ (bsz, E, src) -> (bsz, tgt, src)
    attn = matmul(q, k.swapaxes(-2, -1))
    # attention mask
    if attn_mask is not None:
        attn += attn_mask
    # softmax
    attn = softmax(attn, -1)
    # dropout
    if dropout_p > 0.0:
        attn = dropout(attn, training=training, p=dropout_p)
    # (bsz, tgt, src) @ (bsz, src, E) -> (bsz, tgt, E)
    output = matmul(attn, v)
    return output, attn


def _mha_precheck(query, key, value, key_padding_mask, attn_mask, num_heads, embed_dim_to_check,
                  use_separate_proj_weight, in_proj_bias, q_proj_weight, k_proj_weight, v_proj_weight, bias_k, bias_v):
    tgt_len, bsz, Eq = query.shape
    src_len, _, Ek = key.shape
    _, _, Ev = value.shape

    # shape check
    if query.dim() == 3:
        # Batched Inputs
        assert key.dim() == 3 and value.dim() == 3, ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
                                                     f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() == 3, ("For batched (3-D) `query`, expected `attn_mask` to be `None` or 3-D"
                                          f" but found {attn_mask.dim()}-D tensor instead")
            assert attn_mask.dtype.type in _floating_point or attn_mask.dtype == bool, f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
    else:
        raise AssertionError(f"query should be batched 3D tensor but received {query.dim()}-D query tensor")

    assert Eq == embed_dim_to_check, f"was expecting embedding dimension of {embed_dim_to_check}, but got {Eq}"
    assert Eq % num_heads == 0, f"embed_dim {Eq} not divisible by num_heads {num_heads}"

    if use_separate_proj_weight:
        # allow Multi Head Attention to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[
                                :2], f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        assert q_proj_weight.shape == (
            Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {q_proj_weight.shape}"
        assert k_proj_weight.shape == (
            Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {k_proj_weight.shape}"
        assert v_proj_weight.shape == (
            Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {v_proj_weight.shape}"

        if in_proj_bias is not None:
            separate_bias_shape = in_proj_bias.shape[0] // 3
            assert separate_bias_shape == Eq, f"expecting query, key and value bias shape of {(Eq,)}, but got {(separate_bias_shape,)}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    assert (bias_k is None) == (bias_v is None), 'bias_k and bias_v should either both existed or both be None.'


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k,
                                 bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True,
                                 key_padding_mask=None, need_weights=True, attn_mask=None,
                                 use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
                                 v_proj_weight=None, average_attn_weights=True):
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads

    # precheck
    _mha_precheck(query, key, value, key_padding_mask, attn_mask, num_heads, embed_dim_to_check,
                  use_separate_proj_weight, in_proj_bias, q_proj_weight, k_proj_weight, v_proj_weight, bias_k, bias_v)

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # add bias along batch dimension (currently second)
    # bias_k and bias_v shape: (1, 1, embed_dim)
    if bias_k is not None and bias_v is not None:
        k = cat([k, bias_k.repeat(1, bsz, 1)])  # k shape: cat([ (src, bsz, emb), (1, bsz, emb) ])
        v = cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:  # pad last dimension: left 0 right 1
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # reshape q, k, v for multihead attention and make em batch first
    # q shape: (tgt,bsz,emb)->(tgt,bsz*num_head,head_dim)->(bsz*num_head,tgt,head_dim)
    q = q.contiguous().view((tgt_len, bsz * num_heads, head_dim)).swapaxes(0, 1)
    k = k.contiguous().view((k.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)
    v = v.contiguous().view((v.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        # (bsz*num_head,src,head_dim)->(bsz*num_head,src+1,head_dim)
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        xp = cp if k.data.__class__ is cparray else np
        k = cat([k, Tensor(xp.zeros(zero_attn_shape, dtype=k.dtype), copy=False)], dim=1)
        v = cat([v, Tensor(xp.zeros(zero_attn_shape, dtype=v.dtype), copy=False)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz, src_len), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"

        key_padding_mask = key_padding_mask.view((bsz, 1, 1, src_len)).expand(-1, num_heads, -1, -1).reshape(
            bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == bool:
        new_attn_mask = zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # calculate attention and out projection
    # attn_output: (bsz*num_head, tgt, E); attn_output_weights: (bsz*num_head, tgt, src)
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, training, dropout_p)
    # (bsz*num_head,tgt,head)->(tgt,bsz*num_head,head)->(tgt,bsz,head*num_head)==(tgt,bsz,embed_dim)
    attn_output = attn_output.swapaxes(0, 1).contiguous().view((tgt_len, bsz, embed_dim))
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view((bsz, num_heads, tgt_len, src_len))
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1, keepdim=False) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None
