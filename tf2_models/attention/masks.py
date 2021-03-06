"""Masks to be used when doing attention."""
import tensorflow as tf


def create_look_ahead_mask(size):
    """Create a look ahead mask.
    The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which
    entries should not be used.

    This means that to predict the third word, only the first and second word will be used.
    Similarly to predict the fourth word, only the first, second and the third word will be used and so on.

    :param size: size of max to produce.
    :return:
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    """Create a padding mask.
    Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input.
    The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.

    Explain: Not all sentences are of the same size, therefore there will be vectors with padded zeros. In order to not
    take them into account during attention, we perform masking is those locations.

    :param seq: the input batch of sequences.
    :return: padded mask.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)