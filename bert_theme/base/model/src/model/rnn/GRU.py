from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell
from tensorflow.nn import dynamic_rnn
from tensorflow.nn.rnn_cell import GRUCell
import tensorflow as tf

from src.utils import ConfigUtil
from src.utils.TensorUtil import create_initializer


def GRULayer(input_tensor, num_layers=1):
    input_tensor = tf.reshape(input_tensor, shape=[-1, ConfigUtil.seq_length, ConfigUtil.hidden_size])
    cell = GRUCell(num_units=128, kernel_initializer=create_initializer())
    cell = DropoutWrapper(cell, output_keep_prob=(1 - ConfigUtil.dropout_prob))
    cell = MultiRNNCell([cell] * num_layers) if num_layers > 1 else cell
    outputs, state = dynamic_rnn(cell, input_tensor, dtype=tf.float32)
    return outputs
