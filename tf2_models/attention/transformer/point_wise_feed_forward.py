import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff):
    """

    :param d_model:
    :param dff:
    :return:
    """
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def point_wise_1d_conv(num_kernels_last_layer, num_kernels_first_layer):
    """point_wise_feed_forward_network implemented with 1d convnet.

    :param num_kernels_first_layer:
    :param num_kernels_last_layer:
    :return:
    """
    return tf.keras.Sequential([tf.keras.layers.Conv1D(filters=num_kernels_first_layer, kernel_size=1,
                                                       activation='relu'),
                                tf.keras.layers.Conv1D(filters=num_kernels_last_layer, kernel_size=1)])