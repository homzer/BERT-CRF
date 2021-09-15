import tensorflow as tf

from bert_theme.base.model.src.utils.TensorUtil import gelu, get_activation, create_initializer


def Conv2D(input_tensor, filter_shape, name):
    """
    2D Convolution Layer. Apply relu activation function.
    :param name: scope name.
    :param input_tensor: shape of [batch_size, width, height], or [batch_size, width, height, in_channels]
    :param filter_shape: List, [width of filter, height of filter, in_channels, out_channels].
    For example: filter_shape = [3, 3, 1, 32]
    :return `Tensor` with shape of [batch_size, width, height, out_channels]
    """
    assert len(filter_shape) == 4
    in_channels = filter_shape[2]
    height = input_tensor.shape[1]
    width = input_tensor.shape[2]
    input_tensor = tf.reshape(input_tensor, shape=[-1, height, width, in_channels])
    conv_filter = tf.get_variable(name=name, shape=filter_shape, initializer=create_initializer())
    conv_output = tf.nn.conv2d(input_tensor, conv_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv_output = gelu(conv_output)
    return conv_output


def Pooling2D(input_tensor, filter_size):
    """
    Apply 2-D pooling function
    :param input_tensor: 4-D `Tensor` with shape of [batch_size, width, height, in_channels]
    :param filter_size: must be a list with length of 2.
    e.g. filter_size = [2, 2]
    :return: [batch_size, width / filter_size[0], height / filter_size[1], in_channels]
    """
    assert input_tensor.shape.ndims == 4
    kernel_size = [1, filter_size[0], filter_size[1], 1]
    strides = [1, filter_size[0], filter_size[1], 1]
    return tf.nn.max_pool(input_tensor, kernel_size, strides, padding='SAME')


def Flatten2D(input_tensor, units):
    """
    Apply 2-D Flatten Layer after CNN layers,
    reshape the output of `CNN` to the input of `DNN`
    :param input_tensor: `Tensor` of shape [batch_size, width, height, in_channels]
    :param units: `int` the number of units in `DNN`
    :return: [batch_size, units]
    """
    assert input_tensor.shape.ndims == 4
    width = int(input_tensor.shape[1])
    height = int(input_tensor.shape[2])
    in_channels = int(input_tensor.shape[3])
    input_tensor = tf.reshape(input_tensor, shape=[-1, width * height * in_channels])
    flatten_output = tf.layers.dense(
        inputs=input_tensor,
        units=units,
        activation=get_activation("gelu"),
        kernel_initializer=create_initializer())
    return flatten_output
