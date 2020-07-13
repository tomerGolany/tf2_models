from tf2_models.attention.transformer import point_wise_feed_forward
from tf2_models.attention.transformer import transformer
from tf2_models.attention import masks
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
MAX_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64
EPOCHS = 20


def viz_point_wise_feed_forward():
    sample_ffn = point_wise_feed_forward.point_wise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)

    sample_conv = point_wise_feed_forward.point_wise_1d_conv(512, 1000)
    print(sample_conv(tf.random.uniform((64, 50, 512))).shape)

    print(sample_conv == sample_ffn)


def download_demo_data():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

    sample_string = 'Transformer is awesome.'

    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string

    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    # it = iter(train_examples)
    # ne = next(it)
    # print(ne)

    return train_examples, val_examples, tokenizer_en, tokenizer_pt


train_examples, val_examples, tokenizer_en, tokenizer_pt = download_demo_data()


def encode(lang1, lang2):
    """Add a start and end token to the input and target."""
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2


def tf_encode(pt, en):
    """Wrapper for tf.py_function.
        The tf.py_function will pass regular tensors (with a value and a .numpy()
        method to access it), to the wrapped python function
    """
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en


def filter_max_length(x, y, max_length=MAX_LENGTH):
    """To keep this example small and relatively fast, drop examples with a length of over 40 tokens."""
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def create_dataset():
    train_preprocessed = (
        train_examples
            .map(tf_encode)
            .filter(filter_max_length)
            # cache the dataset to memory to get a speedup while reading from it.
            .cache()
            .shuffle(BUFFER_SIZE))

    val_preprocessed = (
        val_examples
            .map(tf_encode)
            .filter(filter_max_length))

    #
    # Pad and batch examples togheter:
    #
    train_dataset = (train_preprocessed
                     .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
                     .prefetch(tf.data.experimental.AUTOTUNE))

    val_dataset = (val_preprocessed
                   .padded_batch(BATCH_SIZE, padded_shapes=([None], [None])))

    pt_batch, en_batch = next(iter(val_dataset))
    print("pt batch:\nshape: {}\n values: {}\n en batch:shape: {}\n values: {}\n".format(pt_batch.shape, pt_batch,
                                                                                         en_batch.shape, en_batch))
    return train_dataset, val_dataset


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_translation_model():
    """Train Portuguese-to-english translation model.

        from the official tf tutorial.
    :return:
    """

    train_dataset, val_dataset = create_dataset()

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(d_model)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    transformer_obj = transformer.Transformer(num_layers, d_model, num_heads, dff,
                                              input_vocab_size, target_vocab_size,
                                              pe_input=input_vocab_size,
                                              pe_target=target_vocab_size,
                                              rate=dropout_rate)

    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = masks.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = masks.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = masks.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = masks.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer_obj,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        # logger.info("inp: {}\n shape: {}\n".format(inp, inp.shape))
        # tf.print(inp)
        # logger.info("tar: {}\n shape: {}\n".format(tar, tar.shape))
        # tf.print(tar)
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        # logger.info("tar_inp: {}\n shape: {}\n".format(tar_inp, tar_inp.shape))
        # tf.print(tar_inp)
        # logger.info("tar_real: {}\n shape: {}\n".format(tar_real, tar_real.shape))
        # tf.print(tar_real)

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer_obj(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer_obj.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer_obj.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


# def evaluate(inp_sentence):
#     start_token = [tokenizer_pt.vocab_size]
#     end_token = [tokenizer_pt.vocab_size + 1]
#
#     # inp sentence is portuguese, hence adding the start and end token
#     inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
#     encoder_input = tf.expand_dims(inp_sentence, 0)
#
#     # as the target is english, the first word to the transformer should be the
#     # english start token.
#     decoder_input = [tokenizer_en.vocab_size]
#     output = tf.expand_dims(decoder_input, 0)
#
#     for i in range(MAX_LENGTH):
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
#             encoder_input, output)
#
#         # predictions.shape == (batch_size, seq_len, vocab_size)
#         predictions, attention_weights = transformer(encoder_input,
#                                                      output,
#                                                      False,
#                                                      enc_padding_mask,
#                                                      combined_mask,
#                                                      dec_padding_mask)
#
#         # select the last word from the seq_len dimension
#         predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
#
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
#
#         # return the result if the predicted_id is equal to the end token
#         if predicted_id == tokenizer_en.vocab_size + 1:
#             return tf.squeeze(output, axis=0), attention_weights
#
#         # concatentate the predicted_id to the output which is given to the decoder
#         # as its input.
#         output = tf.concat([output, predicted_id], axis=-1)
#
#     return tf.squeeze(output, axis=0), attention_weights


# create_dataset()


# logging.basicConfig(level=logging.INFO)
train_translation_model()