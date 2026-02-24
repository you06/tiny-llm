import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.freqs = 1.0 / (base ** (2 * mx.arange(0, dims // 2) / dims))

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        x_pairs = x.reshape(x.shape[:-1] + (x.shape[-1] // 2, 2))
        if offset is None:
            offset = 0
        else:
            offset = offset.start
        l = mx.arange(0, x.shape[1]) + offset
        freqs = mx.outer(l, self.freqs)
        sin_freqs = mx.sin(freqs)[None, :, None, :]
        cos_freqs = mx.cos(freqs)[None, :, None, :]
        sin_cos_freqs = mx.array([sin_freqs, cos_freqs]).transpose(1, 2, 3, 4, 0)
        cos_sin_freqs = mx.array([cos_freqs, sin_freqs]).transpose(1, 2, 3, 4, 0)
        sin_cos_pairs = x_pairs * sin_cos_freqs # [x[2*i] * sin_freqs[pos, i], x[2*i+1] * cos_freqs[pos, i]]
        cos_sin_pairs = x_pairs * cos_sin_freqs # [x[2*i] * cos_freqs[pos, i], x[2*i+1] * sin_freqs[pos, i]]

        result2i = cos_sin_pairs[..., 0] - cos_sin_pairs[..., 1]
        result2i1 = sin_cos_pairs[..., 0] + sin_cos_pairs[..., 1]
        output = mx.stack([result2i, result2i1], axis=-1)
        output = output.reshape(x.shape)
        return output
