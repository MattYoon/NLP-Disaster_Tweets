# Reference: https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html

import tensorflow as tf
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2))) / tf.cast(d_model, tf.float32)  # formula from the paper
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads=self.get_angles(position = tf.range(position, dtype=tf.float32)[:, tf.newaxis], i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], d_model=d_model)

        sins = tf.math.sin(angle_rads[:, 0::2])  # if index is even
        cosins = tf.math.sin(angle_rads[:, 1::2])  # if index is odd

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sins
        angle_rads[:, 1::2] = cosins

        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]  # add with the original embeddings


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)  # scaling

    if mask is not None:
        logits += (mask * -1e9)  # add negative value to paddings, so after softmax they obtain values close to 0

    attention_weights = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_weights, v)  # (batch_size, num_heads, sent_len, d_model/num_heads)


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


class MultiheadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model # shape of embedding, input of encoder, output of encoder

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.q_dense = tf.keras.layers.Dense(d_model)  # pass through the weights, Wq
        self.k_dense = tf.keras.layers.Dense(d_model)
        self.v_dense = tf.keras.layers.Dense(d_model)

        self.final_dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs, mask):
        batch_size = tf.shape(inputs)[0]

        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # reshape to original
        attention = self.final_dense(attention)

        return attention

print("Encoder Imported")
