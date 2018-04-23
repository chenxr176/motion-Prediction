import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import ReadData
import os
from tensorflow.contrib import rnn
import util as utl



x = tf.placeholder("float64",[2,2])
y = tf.placeholder(tf.float64, [2,2])

Use_to_train = True

Need_to_restore = True

wb1 = {
        'weights': tf.Variable(tf.random_normal([2,1], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[2 ],dtype=tf.float64),dtype=tf.float64),
    }
wb2 = {
        'weights': tf.Variable(tf.random_normal([1, 2], seed=1,dtype=tf.float64),dtype=tf.float64),
        'biases': tf.Variable(tf.constant(0.1, shape=[2, 2],dtype=tf.float64),dtype=tf.float64),
    }

x1 = [[1,2],[3,4]]
y1 = [[4,3],[2,1]]

p = tf.matmul(x,wb1['weights'])
yy = tf.matmul(p,wb2['weights'])


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



with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    start = global_step.eval()
    step = 0

    ckpt_dir = "/home/cxr/test"
    if  Need_to_restore:
        if restore(ckpt_dir+"/"):
            print "restore_seccessfully"
            if not Use_to_train:

                pre = sess.run(yy,feed_dict={
                    x: x1,y:y1,
                })
                step += 1
            else:
                for epoc in range(20):

                    val, l = sess.run(yy, feed_dict={
                            x: x1,
                            y: y1,
                    })
                    if step % 20 == 0:
                         print l
                    if step % 200 == 0:
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                            global_step.assign(step).eval()
                            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
        else:
            print "restore failed"
    else:
        for epoc in range(100):
            print "tarining Epochs = ", epoc
            i=0
            if step % 20 == 0:
                print step
            if step % 50 == 0:
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                    global_step.assign(step).eval()
                    saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
                step += 1
                i+=1






