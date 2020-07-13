import tensorflow as tf
from tf2_models.attention.transformer import encoder
from tf2_models.attention.transformer import decoder
from tf2_models.attention.transformer import transformer


def test_encoder():
    sample_encoder = encoder.Encoder(num_layers=2, d_model=512, num_heads=8,
                                     dff=2048, input_vocab_size=8500,
                                     maximum_position_encoding=10000)

    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
    return sample_encoder_output


def test_decoder():
    sample_encoder_output = test_encoder()
    sample_decoder = decoder.Decoder(num_layers=2, d_model=512, num_heads=8,
                                     dff=2048, target_vocab_size=8000,
                                     maximum_position_encoding=5000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)


sample_transformer = transformer.Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048,
    input_vocab_size=8500, target_vocab_size=8000,
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                               enc_padding_mask=None,
                               look_ahead_mask=None,
                               dec_padding_mask=None)

print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
