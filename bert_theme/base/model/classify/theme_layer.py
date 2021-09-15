import tensorflow as tf
from bert_theme.base.model.bert import shape_helper


def mask_embeddings(embeddings, input_mask):
    """
     根据input_mask对embeddings后面部分置0
    :param embeddings: [batch_size, seq_length, embedding_size]
    :param input_mask: [batch_size, seq_length] 1为有效，0为无效
    :return: [batch_size, seq_length, embedding_size]
    """
    shape = shape_helper.get_shape_list(embeddings, expected_rank=3)
    embedding_size = shape[-1]
    embeddings = tf.reshape(embeddings, shape=[-1, embedding_size])  # [batch_size * seq_length, embedding_size]
    input_mask = tf.cast(tf.reshape(input_mask, shape=[-1, 1]), tf.float32)
    embeddings = tf.reshape(embeddings * input_mask, shape=shape)
    return embeddings


def max_and_mean_concat(embeddings, input_mask):
    """
    根据掩码计算embeddings最后一维的平均值和最大值，并将其连接
    :param embeddings: [batch_size, seq_length, embedding_size]
    :param input_mask: [batch_size, seq_length] 1 为有效， 0 为无效
    :return: embeds_mix [batch_size, embedding_size * 2]
    """
    input_mask = tf.cast(input_mask, dtype=tf.float32)
    lengths = tf.reduce_sum(input_mask, axis=-1, keepdims=True)  # [batch_size, 1]
    # 根据掩码对 embeddings 后面不需要部分置零
    embeddings = embeddings * tf.expand_dims(input_mask, axis=-1)
    # 求和取平均
    embeds_mean = tf.reduce_sum(embeddings, axis=1) / lengths  # [batch_size, embedding_size]
    # 求最大值
    embeds_max = tf.reduce_max(embeddings, axis=1)  # [batch_size, embedding_size]
    # 交叉连接
    embeds_mean = tf.expand_dims(embeds_mean, axis=-1)
    embeds_max = tf.expand_dims(embeds_max, axis=-1)
    embeds_mix = tf.concat([embeds_mean, embeds_max], axis=-1)  # [batch_size, embedding_size, 2]
    shape = shape_helper.get_shape_list(embeds_mix, expected_rank=3)
    embeds_mix = tf.reshape(embeds_mix, shape=[shape[0], -1])
    return embeds_mix


class ThemeLayer(object):
    def __init__(self,
                 embedded_chars,
                 num_themes,
                 initializers,
                 themes,
                 input_mask=None,
                 dropout_rate=0.1,
                 seq_length=512,
                 is_training=True,
                 kernel_size=8,
                 num_kernels=32):
        """
        :param embedded_chars: [batch_size, seq_length, embedding_size]
        :param num_themes: number of different theme types
        :param initializers:
        :param themes: [batch_size, num_classes]
        :param input_mask: [batch_size, seq_length]
        :param dropout_rate:
        :param seq_length:
        :param is_training:
        """
        self.embedded_chars = embedded_chars
        self.num_themes = num_themes
        self.initializers = initializers
        self.themes = themes
        self.dropout_rate = dropout_rate
        self.seq_length = seq_length
        self.is_training = is_training
        self.input_mask = input_mask
        self.embedding_dims = embedded_chars.shape[-1].value
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

    def output_layer(self, name=None):
        concat_embeds = max_and_mean_concat(self.embedded_chars, self.input_mask)  # [batch_size, embedding_size * 2]
        if self.is_training:
            concat_embeds = tf.nn.dropout(concat_embeds, self.dropout_rate)
        with tf.variable_scope("theme" if not name else "theme_" + name):
            with tf.variable_scope("logits"):
                # concat_width = int(concat_embeds.shape[1])
                w = tf.get_variable(
                    'w', shape=[self.embedding_dims * 2, self.num_themes],
                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b = tf.get_variable(
                    'b', shape=[self.num_themes], dtype=tf.float32,
                    initializer=tf.zeros_initializer())
                logits = tf.tanh(tf.nn.xw_plus_b(concat_embeds, w, b))  # [batch_size, num_themes]
                # 获取最大下标，得到预测值
                pred = tf.argmax(logits, -1)
            if self.themes is None:
                return None, pred, logits
            with tf.variable_scope("loss"):
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=self.themes))
        return loss, pred, logits
