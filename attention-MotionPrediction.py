import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import util as utl
import ReadData
import os


PAD = 0
EOS = np.zeros([132])
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 256
decoder_hidden_units = 256
batch_size = 1
n_input = 132
fc1 = 256
sequenceLength = 20
loss_track = []

def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
            for _ in range(batch_size)
        ]


batches = random_sequences(length_from=3, length_to=10,
                           vocab_lower=2, vocab_upper=10,
                           batch_size=batch_size)

'''
def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths
'''

'''
    wb16 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float32),
    }
    wb17 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float64),
                               dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float64),
    }
    wb18 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float64),
                               dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float64),
    }
    wb19 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float64),
                               dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float64),
    }
'''

'''
    wb16 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float64),
                               dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float64),
    }
    wb17 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float64),
                               dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float64),
    }
    wb18 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float64),
                               dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float64),
    }
    wb19 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float64),
                               dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float64), dtype=tf.float64),
    }
'''

train_graph = tf.Graph()
with train_graph.as_default():
    encoder_inputs_raw = tf.placeholder(shape=(None, None, None), dtype=tf.float32, name='encoder_inputs_raw')
    decoder_inputs_raw = tf.placeholder(shape=(None, None, None), dtype=tf.float32, name='decoder_inputs_raw')
    decoder_targets_raw = tf.placeholder(shape=(None, None, None), dtype=tf.float32, name='decoder_targets_raw')

    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    wb11 = {
        'weights': tf.Variable(tf.random_normal([n_input, fc1], seed=1, dtype=tf.float32), dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
            }
    wb12 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
        }
    wb13 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
        }
    wb14 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
        }
    wb15 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
        }



    wbd1 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32), dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
    }
    wbd2 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
    }
    wbd3 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
    }
    wbd4 = {
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
    }
    wbd5 = {
        'weights': tf.Variable(tf.random_normal([fc1, n_input], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_input, ], dtype=tf.float32), dtype=tf.float32),
    }

    x_in = tf.einsum('ijk,kl->ijl', encoder_inputs_raw, wb11['weights'])
    # x_in = tf.sigmoid(x_in)
    x_in = tf.einsum('ijk,kl->ijl', x_in, wb12['weights'])
    # x_in = tf.sigmoid(x_in)
    x_in = tf.einsum('ijk,kl->ijl', x_in, wb13['weights'])
    #x_in = tf.sigmoid(x_in)
    # x_in = tf.sigmoid(x_in)
    x_in = tf.einsum('ijk,kl->ijl', x_in, wb14['weights'])
    # x_in = tf.sigmoid(x_in)
    x_in = tf.einsum('ijk,kl->ijl', x_in, wb15['weights'])
    #x_in = tf.sigmoid(x_in)

    encoder_inputs_fc = x_in

    attention = tf.Variable(tf.zeros([fc1, fc1]), dtype=tf.float32)

    # embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

    # encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_fc,
        dtype=tf.float32, time_major=True,
    )
    N = tf.einsum('ijk,kl->ijl', encoder_outputs, attention)

    # decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    # encoder_outputs

    # n = tf.Variable(tf.zeros([1,sequenceLength,132],dtype=tf.float32),trainable=False,dtype=tf.float32)
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    # att_d_e=decoder_inputs_embedded+N

    x_din = tf.einsum('ijk,kl->ijl', decoder_inputs_raw, wb11['weights'])
    # x_in = tf.sigmoid(x_in)
    x_din = tf.einsum('ijk,kl->ijl', x_din, wb12['weights'])
    # x_in = tf.sigmoid(x_in)
    x_din = tf.einsum('ijk,kl->ijl', x_din, wb13['weights'])
    #x_din = tf.sigmoid(x_din)
    # x_in = tf.sigmoid(x_in)
    x_din = tf.einsum('ijk,kl->ijl', x_din, wb14['weights'])
    # x_in = tf.sigmoid(x_in)
    x_din = tf.einsum('ijk,kl->ijl', x_din, wb15['weights'])
    #x_din = tf.sigmoid(x_din)


    att_d_e = x_din + N
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, att_d_e,
        initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope="plain_decoder",
    )
    y_out = tf.einsum('ijk,kl->ijl', decoder_outputs, wbd1['weights'])
    y_out = tf.einsum('ijk,kl->ijl', y_out, wbd2['weights'])
    y_out = tf.einsum('ijk,kl->ijl', y_out, wbd3['weights'])
    #y_out = tf.sigmoid(y_out)
    y_out = tf.einsum('ijk,kl->ijl', y_out, wbd4['weights'])
    y_out = tf.einsum('ijk,kl->ijl', y_out, wbd5['weights'])
    #y_out = tf.sigmoid(y_out)
    # decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    # decoder_prediction = tf.argmax(decoder_logits, 2)
    # stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    #    logits=decoder_logits,
    # )
    pred = tf.reshape(y_out, [-1, n_input])

    label = tf.reshape(decoder_targets_raw, [-1, n_input])

    loss = tf.reduce_mean(tf.pow(pred - label, 2))
    train_op = tf.train.AdamOptimizer().minimize(loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()

epochs = 20000

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    start = global_step.eval()
    ckpt_dir = "/home/cxr/BvhLstm1-2"
    filename = "/home/cxr/7-2"
    for epoch in range(epochs):
        print "tarining Epochs = ", epoch
        r = ReadData.Actionreader()
        v ,_= utl.readData(filename)
        length = len(v)
        i = 0
        step = 0

        batch_xs, batch_ys = utl.get_batch(i, v, sequenceLength, batch_size)

        while batch_xs!=None and batch_ys!=None:
            _,losss = sess.run([train_op,loss], feed_dict={
                encoder_inputs_raw:  batch_xs,
                decoder_targets_raw: batch_ys,
                decoder_inputs_raw: batch_ys,
            })
            if step % 20 == 0:
                print losss
            if step % 200 == 0:
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                    global_step.assign(step).eval()
                    saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
            step += 1
            i += 1
            batch_xs, batch_ys = utl.get_batch(i, v, sequenceLength, batch_size)
            loss_track.append(losss)

plt.plot(loss_track)
plt.show()

