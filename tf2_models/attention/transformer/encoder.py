"""Encoder class of a transformer."""
import tensorflow as tf
from tf2_models.attention import layers
from tf2_models.attention.transformer import point_wise_feed_forward
from tf2_models.attention.transformer import positional_encoding


class EncoderLayer(tf.keras.layers.Layer):
    """Single Encoder layer as described in the paper "Attention is all you need"
        The layer has two sub-layers:
        1. The first is a multi-head self-attention mechanism,
        2. second is a simple, position wise fully connected feed-forward network.

        residual connection is applied around each of the two sub-layers, followed by layer normalization.
        The output of each sub-layer is LayerNorm(x + Sublayer(x))

        In the paper, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension
        d_model = 512.

    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """Initialize an Encoder layer.

        :param d_model: all sub-layers in the model, as well as the embedding layers, produce outputs of dimension
        d_model
        :param num_heads: Number of heads in the multi-head self-attention layer. (8 in the paper.)
        :param dff: Number of Neurons in the inner fully connected network. (2048 in the paper.)
        :param rate: dropout rate.
        """
        super(EncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """

        :param x: Input to the the encoder. shape: [batch_size, num_words, d_model]
        :param training: training mode or inference.
        :param mask: mask matrix for the attention layer.
        :return: output from a single encoder layer - [batch_size, num_words, d_model]
        """

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class Encoder(tf.keras.layers.Layer):
    """The full Encoder component.

    The encoder is composed of a stack of N = 6 Encoder layers.

    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        """Initialize an Encoder object.

        :param num_layers: Number of encoder layers. (6 in the paper.)
        :param d_model: all sub-layers in the model, as well as the embedding layers, produce outputs of dimension
        d_model
        :param num_heads:  Number of heads in the multi-head self-attention layer. (8 in the paper.)
        :param dff: Number of Neurons in the inner fully connected network. (2048 in the paper.)
        :param input_vocab_size: (In NLP case.) to create the emabadding.
        :param maximum_position_encoding:
        :param rate:
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding.positional_encoding(maximum_position_encoding,
                                                                    self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Call the Encoder component for a language model case.

        :param x: Input batch of sentences -> [batch_size, num_of_words, 1 or voc size.]
        :param training:
        :param mask: optional.
        :return: encoded words. (batch_size, input_seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scaling ?
        x += self.pos_encoding[:, :seq_len, :]  # Adding positional encoding.

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
