from tensorflow.contrib.layers import layer_norm


def LayerNorm(input_tensor):
    """Run layer normalization on the last dimension of the tensor."""
    return layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1)
