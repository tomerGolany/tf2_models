import tensorflow as tf
from tf2_models.attention import layers
from tf2_models.attention.transformer import point_wise_feed_forward
from tf2_models.attention.transformer import positional_encoding


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer as described in the paper.

    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """Initialize DecoderLayer object.

        :param d_model: number of features through all the model. from the embedding to the output.
        :param num_heads:
        :param dff:
        :param rate:
        """
        super(DecoderLayer, self).__init__()

        self.mha1 = layers.MultiHeadAttention(d_model, num_heads)
        self.mha2 = layers.MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """

        :param x: input to the decoder. This is the previous output.
        :param enc_output: the encoded words from the encoder - [batch_size, num_words, d_model]
        :param training: mode.
        :param look_ahead_mask: the mask for the attention.
        :param padding_mask:
        :return:  shape (batch_size, input_seq_len, d_model)
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        #
        # First attention layer in the decoder. The mask will actually be combined:
        #
        # It is used to pad and mask future tokens in the input received by the decoder.
        #
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        #
        # This padding mask is used to mask the encoder outputs.
        #
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    """The full Decoder component.

    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        """Initialize Decoder object.

        :param num_layers: Number of decoder layers stacked.
        :param d_model: number of features through all the model. from the embedding to the output.
        :param num_heads:
        :param dff:
        :param target_vocab_size:
        :param maximum_position_encoding:
        :param rate:
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        """

        :param x: Input to the decoder. during training it is the target labels, and during inference? TODO.
        :param enc_output: The output from the encoder. the encoded words. [batch_size, num_words, d_model]
        :param training: mode.
        :param look_ahead_mask:
        :param padding_mask:
        :return:
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights