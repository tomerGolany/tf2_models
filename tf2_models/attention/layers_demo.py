import numpy as np
from tf2_models.attention import layers
import tensorflow as tf
from tf2_models.attention import masks


def print_out(q, k, v):
  temp_out, temp_attn = layers.scaled_dot_product_attention(
      q, k, v, None)
  print('Attention weights are:')
  print(temp_attn)
  print('Output is:')
  print(temp_out)


def demo():
    # 4 words, dim_k is 3
    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    # This query aligns with a repeated key (third and fourth),
    # so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    # This query aligns equally with the first and second key,
    # so their values get averaged.
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    print_out(temp_q, temp_k, temp_v)

    #
    # Check Keras builtin implementation:
    #
    print("Now performing attention with keras:")
    att = tf.keras.layers.Attention(use_scale=True)
    print(att)
    print(att([temp_q, temp_v, temp_k]))

    mask = masks.create_look_ahead_mask(4)
    temp_q = tf.constant([[0, 10, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)  # (4, 3)
    print(mask)
    print("Adding look ahead mask:")
    temp_out, temp_attn = layers.scaled_dot_product_attention(temp_q, temp_k, temp_v, mask)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

# demo()


def multi_head_demo():
    temp_mha = layers.MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape, attn.shape)

multi_head_demo()