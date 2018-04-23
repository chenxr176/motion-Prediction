import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import ReadData
import os
from tensorflow.contrib import rnn
import util as utl

learning_rate = 0.001
training_epochs = 3000
batch_size = 1
display_step = 1
classnum = 18
partnum = 27
examples_to_show = 10

training_iters = 10000
n_sequence = 10
n_hidden_1 = 512
n_hidden_2 = 128
#n_input = (classnum+1)*partnum
n_input=132
n_hidden_1_2 = n_hidden_1 / 2
n_hidden_1_3 = n_hidden_1 / 3

p_keep_conv = 0.7

cell_num = 3
NUM_LAYERS=3


x = tf.placeholder("float64", [None, n_sequence, n_input])
y = tf.placeholder(tf.float64, [None, n_input])

Use_to_train = True

Need_to_restore = True
'''
weights={
    'encoder_h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_input])),
}

biases={
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}
'''
with tf.variable_scope("input_full"):
    wb11={
        'weights': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb12 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb13 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb14 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb15 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb16 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb17 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb18 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb19 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }

with tf.variable_scope("out_full"):
    wb1 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb2 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb3 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb4 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb5 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb6 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb7 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb8 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden_1, ],dtype=tf.float64),dtype=tf.float64),
    }
    wb9 = {
        'weights': tf.Variable(tf.random_normal([n_hidden_1, n_input], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[n_input, ],dtype=tf.float64),dtype=tf.float64),
    }


'''
def Lstm_cell(Hiddensize):
    cell = tf.contrib.rnn.BasicLSTMCell(Hiddensize,state_is_tuple = True, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.7)

def Autocoder_lstm(X):
    Lstm_layer1 = tf.contrib.BasicLSTMCell(weights['encoder_h1'],forget_bias=1.0,state_is_tuple=True)
    init_state=Lstm_layer1.zero_state(batch_size,dtype=tf.float32)
    outputs1,final_state1=tf.nn.dynamic_rnn(Lstm_layer1,X,initial_state=init_state,time_major=False)
    results1=tf.matmul(final_state1[1],weights['encoder_h2'])+biases['encoder_b2']

    Lstm_layer2 = tf.contrib.rnn.BasicLSTMCell(weights['encoder_h2'],forget_bias=1.0,state_is_tuple=True)
    outputs2,final_state2=tf.nn.dynamic_rnn(Lstm_layer2,results1,initial_state=init_state,time_major=False)
    results2=tf.matmul(final_state2[1],weights['decoder_h1'])+biases['decoder_b1']

    Lstm_layer3= tf.contrib.rnn.BasicLSTMCell(weights['decoder_h2'], forget_bias=1.0, state_is_tuple=True)
    outputs3, final_state3 = tf.nn.dynamic_rnn(Lstm_layer2, results1, initial_state=init_state, time_major=False)
    results3 = tf.matmul(final_state2[1], weights['decoder_h1']) + biases['decoder_b1']

    return results2

def CNNmodel(x,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden):
    l1a=tf.nn.relu(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME'))

    l1=tf.nn.max_pool(l1a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    l1=tf.nn.dropout(l1,p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'))

    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'))

    l3 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    l3=tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l1, p_keep_conv)

    l4=tf.nn.relu(tf.matmul(l3,w4))
    l4=tf.nn.dropout(l4,p_keep_hidden)

'''
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = n_hidden_1,forget_bias = 1.0,state_is_tuple = True,reuse = None)

lstm_cell1 = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.8)

init_state = lstm_cell1.zero_state(batch_size,dtype=tf.float64)



def autocoder_lstm(x):
    x = tf.reshape(x, [-1, n_input])
    with tf.variable_scope("input_full"):
        x_in = tf.matmul(x, wb11['weights']) + wb11['biases']
        #x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb12['weights']) + wb12['biases']
        #x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb13['weights']) + wb13['biases']
        #x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb14['weights']) + wb14['biases']
        #x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb15['weights']) + wb15['biases']
        x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb16['weights']) + wb16['biases']
        #x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb17['weights']) + wb17['biases']
        #x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb18['weights']) + wb18['biases']
        #x_in = tf.sigmoid(x_in)
        x_in = tf.matmul(x_in, wb19['weights']) + wb19['biases']
    #with tf.variable_scope("input_full2"):
    #   x_in = tf.matmul(x_in, weight_biases3['weights']) + weight_biases3['biases']
        x_in = tf.reshape(x_in, [-1, n_sequence, n_hidden_1])

    outputs,finalstate = tf.nn.dynamic_rnn(lstm_cell1,x_in,initial_state=init_state,time_major=False)

    out = finalstate[1]

    with tf.variable_scope("out_full"):
        x_out = tf.matmul(out, wb1['weights']) + wb1['biases']
        #x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb1['weights']) + wb1['biases']
        #x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb2['weights']) + wb2['biases']
        #x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb3['weights']) + wb3['biases']
       # x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb4['weights']) + wb4['biases']
        #x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb5['weights']) + wb5['biases']
        x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb6['weights']) + wb6['biases']
        #x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb7['weights']) + wb7['biases']
        #x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb8['weights']) + wb8['biases']
        #x_out = tf.sigmoid(x_out)
        x_out = tf.matmul(x_out, wb9['weights']) + wb9['biases']
    return x_out
    # results = tf.reshape(results,[batch_size,n_sequence,n_input])
    # return results


pred = autocoder_lstm(x)

predic = tf.reshape(pred,[-1,n_input])
#predic = tf.argmax(predic,axis=1)

yo = tf.reshape(y,[-1,n_input])
#y = tf.argmax(y,axis=1)

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predic,labels=yo))

#y = tf.float32(y)
cost=tf.reduce_mean(tf.sqrt(tf.pow(yo-predic,2)))
#cost = cross_entropy
train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
#pred = tf.reshape(pred,[-1,classnum])
#pred = tf.argmax(pred,axis=1)
#y = tf.reshape(y,[-1,-1,classnum])
#y=tf.argmax(y,axis=1)
#cost = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.argmax(y),pred)

correct_pred = tf.equal(predic, yo)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()

def restore(ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess,ckpt.model_checkpoint_path)
        return True
    else :
        return False
def embedding(actionNum,class_num=12):#fanhuiyigezhi
    v =[]
    for i in actionNum:
        num = int(i)
        em = np.zeros([class_num],np.float32)
        em[num-1] = 1
        v.append(em)
    v = np.reshape(v,(-1))
    return v

def readData(filename):
    fo = open(filename,'r')
    line = fo.readline()
    v=[]
    while line:
        n=[]
        num = line.split()
        for i in num:
            n.append(int(i))
        v.append(n[:-1])
        line = fo.readline()
    return v

def get_batch(v,i,lenth,batchsize,n_sequence):
    if i+2>=lenth:
        return None,None
    b_x = []
    b_y = []
    b_x.append(embedding(v[i]))
    b_x.append(embedding(v[i+1]))
    b_y.append(embedding(v[i+2]))
    return b_x,b_y




with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/cxr/tfboard/autocoderlstm", sess.graph)
    sess.run(tf.global_variables_initializer())

    start = global_step.eval()
    step = 0
    r = ReadData.Actionreader()
    ckpt_dir = "/home/cxr/BvhLstm1-2"
    filename = "/home/cxr/7-2"
    if  Need_to_restore:
        if restore(ckpt_dir+"/"):
            print "restore_seccessfully"
            if not Use_to_train:
                r.reset()
                v,timelist=utl.readData(filename)
                length = len(v)
                i=0
                step = 0
                batch_xs, batch_ys = utl.get_batch(v, i, length, batch_size, n_sequence)
                print len(batch_xs)
                while batch_xs and step<=2000:
                    pre = sess.run([predic], feed_dict={
                        x: batch_xs,
                    })
                    r.out_data(pre[0],timelist[step+2],ckpt_dir)

                    #batch_xs = batch_xs[0]
                    #batch_xs = batch_xs[1:]
                    #batch_xs.append(utl.transform(pre,classnum))
                    #batch_xs = [batch_xs]
                    batch_xs, batch_ys = utl.get_batch(v, step, length, batch_size, n_sequence)

                    #print batch_xs
                    step += 1
            else:
                for epoc in range(training_epochs):
                    print "tarining Epochs = ", epoc
                    r = ReadData.Actionreader()
                    v, timelist = utl.readData(filename)
                    length = len(v)
                    i = 0

                    batch_xs, batch_ys = utl.get_batch(v, i, length, classnumber=classnum, batchsize=batch_size,
                                                       n_sequence=n_sequence)
                    # print batch_xs
                    while batch_xs and batch_ys:
                        val, l = sess.run([train_op, cost], feed_dict={
                            x: batch_xs,
                            y: batch_ys,
                        })
                        if step % 20 == 0:
                            print l
                        if step % 200 == 0:
                            if not os.path.exists(ckpt_dir):
                                os.makedirs(ckpt_dir)
                                global_step.assign(step).eval()
                                saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
                        step += 1

                        i += 1
                        batch_xs, batch_ys = utl.get_batch(v, i, length, classnumber=classnum, batchsize=batch_size,
                                                           n_sequence=n_sequence)


        else:
            print "restore failed"
    else:
        for epoc in range(training_epochs):
            print "tarining Epochs = ", epoc

            r =ReadData.Actionreader()
            v,timelist =utl.readData(filename)
            length = len(v)
            i=0

            batch_xs, batch_ys = utl.get_batch(v, i, length,classnumber=classnum, batchsize=batch_size, n_sequence=n_sequence)
            #print batch_xs
            while batch_xs and batch_ys:
                val, l = sess.run([train_op, cost], feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                if step % 20 == 0:
                    print l
                if step % 200 == 0:
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                        global_step.assign(step).eval()
                        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
                step += 1

                i+=1
                batch_xs, batch_ys = utl.get_batch(v, i, length, classnumber=classnum, batchsize=batch_size,
                                                   n_sequence=n_sequence)





