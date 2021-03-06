import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = 20
batch_size = 20
n_input = 132
fc1 = 256


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

    encoder_inputs_raw = tf.placeholder(shape=(None,None),dtype= tf.float32,name = 'inputs_raw')
    decoder_inputs_raw = tf.placeholder(shape=(None,None),dtype=tf.float32,name = 'decoder_inputs_raw')
    decoder_targets_raw = tf.placeholder(shape=(None,None),dtype=tf.float32,name = 'decoder_targets_raw')


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
        'weights': tf.Variable(tf.random_normal([n_input, fc1], seed=1, dtype=tf.float32), dtype=tf.float32),
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
        'weights': tf.Variable(tf.random_normal([fc1, fc1], seed=1, dtype=tf.float32),
                               dtype=tf.float32),
        'biases': tf.Variable(tf.constant(0.1, shape=[fc1, ], dtype=tf.float32), dtype=tf.float32),
    }

    x_in = tf.matmul(encoder_inputs_raw, wb11['weights']) + wb11['biases']
    # x_in = tf.sigmoid(x_in)
    x_in = tf.matmul(x_in, wb12['weights']) + wb12['biases']
    # x_in = tf.sigmoid(x_in)
    x_in = tf.matmul(x_in, wb13['weights']) + wb13['biases']
    x_in = tf.sigmoid(x_in)
    # x_in = tf.sigmoid(x_in)
    x_in = tf.matmul(x_in, wb14['weights']) + wb14['biases']
    # x_in = tf.sigmoid(x_in)
    x_in = tf.matmul(x_in, wb15['weights']) + wb15['biases']
    x_in = tf.sigmoid(x_in)



    encoder_inputs_fc = x_in

    attention = tf.Variable(tf.zeros([input_embedding_size,input_embedding_size]), dtype=tf.float32)

    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)





    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
    )
    N = tf.einsum('ijk,kl->ijl', encoder_outputs,attention)





    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)




    #encoder_outputs

    n = tf.Variable(tf.zeros([1,100,20],dtype=tf.float32),trainable=False,dtype=tf.float32)
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    #att_d_e=decoder_inputs_embedded+N
    att_d_e = decoder_inputs_embedded+tf.concat([N,n],0)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, att_d_e,
        initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope="plain_decoder",
    )

    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    decoder_prediction = tf.argmax(decoder_logits, 2)
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )
    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

loss_track = []
epochs = 3000


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch = next(batches)
        encoder_inputs_, _ = make_batch(batch)
        decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])
        decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])
        feed_dict = {encoder_inputs: encoder_inputs_, decoder_inputs: decoder_inputs_,
                     decoder_targets: decoder_targets_,
                     }
        _, l = sess.run([train_op, loss], feed_dict)
        loss_track.append(l)
        if epoch == 0 or epoch % 1000 == 0:
            print('loss: {}'.format(sess.run(loss, feed_dict)))
            predict_ = sess.run(decoder_prediction, feed_dict)
            for i, (inp, pred) in enumerate(zip(feed_dict[encoder_inputs].T, predict_.T)):
                print('input > {}'.format(inp))
                print('predicted > {}'.format(pred))
                if i >= 20:
                    break

plt.plot(loss_track)
plt.show()
