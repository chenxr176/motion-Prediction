import numpy as np
import tensorflow as tf

def embedding(actionNum,class_num=12):#fanhuiyigezhi
    v =[]
    for i in actionNum:
        num = int(i)
        em = np.zeros([class_num+1],np.float32)
        em[num] = 1
        v.append(em)
    v = np.reshape(v,(-1))
    return v
'''
def readData(filename):
    fo = open(filename,'r')
    line = fo.readline()
    v=[]
    timelist=[]
    classlist=[]
    t=0
    while line:
        n=[]
        num = line.split()
        for i in num:
            n.append(int(i))
        v.append(n[:-1])
        classlist=[n[-1]]
        timelist.append(t)
        line = fo.readline()
        t+=1
    return v,timelist,classlist
'''
def get_batch(i,v,sequenceLength,batch_size):
     lv = len(v)
     lvv = len(v[0])
     n = int(lv/sequenceLength)
     nv = v[0:n*sequenceLength]
     shape = np.shape(nv)

     rv = np.reshape(nv, [shape[0]/sequenceLength, sequenceLength ,lvv])
     if i+batch_size+1<len(rv):
         return rv[i:i+batch_size],rv[i+1:i+batch_size+1]
     else:
         return None,None








def transform(v,classnum):
    ret = []
    for i in v[0]:
        p = np.zeros([classnum])
        p[i] = 1
        ret.append(p)
    ret = np.reshape(ret,(-1))
    return ret

#y = tf.placeholder(tf.float32,[None,33])
#t= tf.reshape(y,[2,3,11])
#t = tf.argmax(t,axis=2)


def readData(filename):
    fo = open(filename,'r')
    line = fo.readline()
    v=[]
    timelist=[]
    t=0
    while line:
        n=[]
        num = line.split()
        for i in num:
            n.append((float)(i))
        v.append(n)
        timelist.append(t)
        line = fo.readline()
        t+=1
    return v,timelist
'''

with tf.Session() as sess:
    v = [[1, 2, 3], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]]

    x, y_in = get_batch(v, 0, len(v), 10, 2, 5)
    y_o  = sess.run(t,feed_dict={
        y: y_in
    })
    print y_o
'''
#v = [[1,2,3],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10]]

#x,y=get_batch(v,0,len(v),10,2,5)
#print x
#print y
# v, ti = readData("/media/cxr/TOSHIBA EXT/7-1")
#print len(v[0])