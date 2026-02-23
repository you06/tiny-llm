import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


# You will also need to implement the linear function in basics.py first.
# For linear, it takes a tensor of the shape N.. x I, a weight matrix of the shape O x I, and a bias vector of the shape O.
# The output is of the shape N.. x O.
# I is the input dimension and O is the output dimension.
def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    out = x @ w.T
    if bias is None:
        return out
    return out + bias


def silu(x: mx.array) -> mx.array:
    pass
