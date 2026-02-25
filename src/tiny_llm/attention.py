import mlx.core as mx
from .basics import softmax, linear


# attention = softmax(QKt * scale + mask) V
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = (query @ key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return softmax(scores, axis=-1) @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    # MultiHead = concat(head 1, ..., head O) WO
    # head i = Attention(QWiQ, KWiK, VWiV)
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        query = linear(query, self.wq)
        key = linear(key, self.wk)
        value = linear(value, self.wv)
        head_size = int(self.hidden_size / self.num_heads)
        query_heads = query.reshape(query.shape[:-1] + (self.num_heads, head_size)).transpose(2, 0, 1, 3)
        key_heads = key.reshape(key.shape[:-1] + (self.num_heads, head_size)).transpose(2, 0, 1, 3)
        value_heads = value.reshape(value.shape[:-1] + (self.num_heads, head_size)).transpose(2, 0, 1, 3)

        attention_heads = [scaled_dot_product_attention_simple(
            query_heads[i],
            key_heads[i],
            value_heads[i],
            mx.rsqrt(self.hidden_size / self.num_heads),
            mask,
        ) for i in range(self.num_heads)]

        attentions = mx.concatenate(attention_heads, axis=-1)
        return linear(attentions, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    # query length = L, key/value length = S
    mask = mx.triu(mx.full((L, S), float("-inf"), dtype=dtype), k = S - L + 1)
    return mask

# attention = softmax(QKt * scale + mask) V
def scaled_dot_product_attention_grouped(
    query: mx.array, # N.. x H_q x L x D
    key: mx.array, # N.. x H x S x D
    value: mx.array, # N.. x H x S x D
    scale: float | None = None,
    mask: mx.array | str | None = None, # N.. x H_q x L x S
) -> mx.array: # N.. x H_q x L x D
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    q = query.shape[-3] // key.shape[-3]
    key = head_grouped(key, q)
    value = head_grouped(value, q)
    scores = (query @ key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            L, S = scores.shape[-2], scores.shape[-1]
            mask = causal_mask(L, S, scores.dtype)
        scores = scores + mask

    return softmax(scores, axis=-1) @ value

def head_grouped(x: mx.array, q: int) -> mx.array:
    if q == 1:
        return x
    ret_shape = x.shape[:-3] + (x.shape[-3] * q,) + x.shape[-2:]
    x = mx.expand_dims(x, axis=-3) # N.. x H x S x D -> N.. x H x 1 x S x D
    x = mx.broadcast_to(x, x.shape[:-3] + (q,) + x.shape[-2:])
    x = x.reshape(ret_shape)
    return x

def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
