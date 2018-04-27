import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import util as utl
import ReadData
import os

learning_rate = 0.001
training_epochs = 500
batch_size = 1
display_step = 1
classnum = 18
partnum = 27

examples_to_show = 10

training_iters = 10000
n_sequence = 10
n_hidden_1 = 1024
n_hidden_2 = 128
n_input = (classnum+1)*partnum
n_hidden_1_2 = n_hidden_1 / 2
n_hidden_1_3 = n_hidden_1 / 3

p_keep_conv = 0.7

cell_num = 3
NUM_LAYERS=3


x = tf.placeholder("float", [None, n_sequence, n_input])
y = tf.placeholder(tf.float32, [None, n_input])

Use_to_train = True
Need_to_restore = None
with tf.variable_scope("input_full1"):

    weights_1={
        'in':tf.Variable(tf.random_normal([n_input,n_hidden_1*2],seed=1),name = "weight1_in"),
        'out':tf.Variable(tf.random_normal([n_hidden_1*2,n_hidden_1],seed=1),name = "weight1_out"),
    }

    biases_1={
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_1*2,]),name = "biases_1_in"),
        'out':tf.Variable(tf.constant(0.1,shape=[n_hidden_1,])),
    }
    with tf.variable_scope("input_full1"):
        weight_bias_2={
            'weights':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_1],seed=1)),
            'biases':tf.Variable(tf.constant(0.1,shape=[n_hidden_1,])),
        }
    with tf.variable_scope("input_full2"):
        weight_bias_3={
            'weights':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_1],seed=1)),
            'biases':tf.Variable(tf.constant(0.1,shape=[n_hidden_1,])),
        }

with tf.variable_scope("out_full"):
    with tf.variable_scope("out_full1"):
        out_all_1 ={
        'weights':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_1])),
        'biases':tf.Variable(tf.constant(0.1,shape=[n_hidden_1,])),
        }
    with tf.variable_scope("out_full2"):
        out_all_2 ={
        'weights':tf.Variable(tf.random_normal([n_hidden_1,n_input])),
        'biases':tf.Variable(tf.constant(0.1,shape=[n_input,])),
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

def Lstm_cell():
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = n_hidden_1,forget_bias = 1.0,state_is_tuple = True)

    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.8)
    return lstm_cell
lstm_cell = tf.contrib.rnn.MultiRNNCell([Lstm_cell() for _ in range(NUM_LAYERS)])
init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)



def autocoder_lstm(x,weight1,biases1,weight_biases2,weight_biases3,out_all1,out_all2):
    x = tf.reshape(x, [-1, n_input])
    with tf.variable_scope("input_full"):
        x_in = tf.matmul(x, weight1['in']) + biases1['in']
        x_in = tf.matmul(x_in, weight1['out']) + biases1['out']
        with tf.variable_scope("input_full1"):
            x_in = tf.matmul(x_in, weight_biases2['weights']) + weight_biases2['biases']
        with tf.variable_scope("input_full2"):
            x_in = tf.matmul(x_in, weight_biases3['weights']) + weight_biases3['biases']
        x_in = tf.reshape(x_in, [-1, n_sequence, n_hidden_1])
    outputs,finalstate = tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state=init_state,time_major=False)

    out = finalstate[1][-1]

    out = tf.reshape(out,[-1,n_hidden_1])
    with tf.variable_scope("out_full"):
        with tf.variable_scope("out_full1"):
            outputsall1 = tf.matmul(out,out_all1['weights'])+out_all1['biases']
        with tf.variable_scope("out_full2"):
            outputsall2 = tf.matmul(outputsall1,out_all2['weights'])+out_all2['biases']

    return outputsall2
    # results = tf.reshape(results,[batch_size,n_sequence,n_input])
    # return results


pred = autocoder_lstm(x, weights_1, biases_1,weight_bias_2,weight_bias_3,out_all_1,out_all_2)
predic = tf.reshape(pred,[-1,27,classnum+1])
#predic = tf.argmax(predic,axis=1)

yo = tf.reshape(y,[-1,27,classnum+1])
#y = tf.argmax(y,axis=1)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predic,labels=yo))

#y = tf.float32(y)
cost=tf.reduce_mean(tf.pow(yo-predic,2))
#cost = cross_entropy
train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(pred, y)
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
    writer = tf.summary.FileWriter("/home/cxr/tfboard/autocoderattention", sess.graph)
    sess.run(tf.global_variables_initializer())

    start = global_step.eval()
    step = 0
    r = ReadData.Actionreader()
    ckpt_dir = "/home/cxr/3lrAutocoder727"
    filename = "/home/cxr/class/StateIn24walking"
    if  Need_to_restore:
        if restore(ckpt_dir):
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
                    r.reset()
                    batch_xs, batch_ys = r.get_batch(batch_size, n_sequence,
                                                     r.ReadFile(filename))

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
                        batch_xs, batch_ys = r.get_batch(batch_size, n_sequence,
                                                         r.ReadFile(filename))

        else:
            print "restore failed"
    else:
        for epoc in range(training_epochs):
            print "tarining Epochs = ", epoc

            r =ReadData.Actionreader()
            v,timelist,classlist =utl.readData(filename)
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





