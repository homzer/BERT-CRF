import tensorflow as tf

from bert_theme.base.model.src.utils.TensorUtil import gelu, get_activation, create_initializer


def Conv1D(input_tensor, filter_shape, name):
    """
    1D Convolution Layer.
    :param input_tensor: shape of [batch_size, width] or [batch_size, width, in_channels]
    :param filter_shape: List, [width of filter, in_channels, out_channels].
    :param name: scope name.
    :return: `Tensor` with shape of [batch_size, width, out_channels]
    """
    assert len(filter_shape) == 3
    in_channels = filter_shape[1]
    width = input_tensor.shape[1]
    input_tensor = tf.reshape(input_tensor, shape=[-1, width, in_channels])
    conv_filter = tf.get_variable(name=name, shape=filter_shape, initializer=create_initializer())
    conv_output = tf.nn.conv1d(input_tensor, conv_filter, stride=1, padding='SAME')
    conv_output = gelu(conv_output)
    return conv_output


def Pooling1D(input_tensor, filter_size):
    """
    Apply 1-D pooling function
    :param input_tensor: 3-D `Tensor` with shape of [batch_size, width, in_channels]
    :param filter_size: An Integer represents the size of the pooling window.
    e.g. filter_size = 2
    :return: [batch_size, width / filter_size, in_channels]
    """
    assert input_tensor.shape.ndims == 3
    return tf.layers.max_pooling1d(
        input_tensor,
        pool_size=filter_size,
        strides=filter_size,
        padding='SAME')


def Flatten1D(input_tensor, units):
    """
    Apply 1-D Flatten Layer after CNN layers,
    reshape the output of `CNN` to the input of `DNN`
    :param input_tensor: `Tensor` of shape [batch_size, width, in_channels]
    :param units: `int` the number of units in `DNN`
    :return: [batch_size, units]
    """
    assert input_tensor.shape.ndims == 3
    width = int(input_tensor.shape[1])
    in_channels = int(input_tensor.shape[2])
    input_tensor = tf.reshape(input_tensor, shape=[-1, width * in_channels])
    flatten_output = tf.layers.dense(
        inputs=input_tensor,
        units=units,
        activation=get_activation("gelu"),
        kernel_initializer=create_initializer())
    return flatten_output
