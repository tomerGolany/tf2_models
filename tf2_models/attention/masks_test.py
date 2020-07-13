from tf2_models.attention import masks
import tensorflow as tf

x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
print("Input seq: shape: {}\n values:\n {}".format(x.shape, x))
padded_mask = masks.create_padding_mask(x)
print("Padding mask: shape: {}\n values:\n {}".format(padded_mask.shape, padded_mask))
