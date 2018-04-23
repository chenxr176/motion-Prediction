import io
import numpy as np
import os
import re
class Actionreader(object):
    datanum =0
    Datapath=""
    FileName ="/home/cxr/subject/subject1_ideal.log"
    outFileDir = "/media/cxr/TOSHIBA EXT/"

    ACCESSMODE='r'
    ACCESSMODEW = 'a'
    FileVECTOR=[]
    ClassLable=[]
    classlable=[]
    ReadQueue = []
    ReadQueueFull = False
    QueueSize = 3
    files = []
    fo = None
    fileCount = 0
    def reset(self):
        Actionreader.fileCount = 0
    def ReadFile(self,dirPath):

        for parent,dirnames,filenames in os.walk(dirPath):
            for filename in filenames:
                if re.search("subject1_ideal.log",filename) :
                    Actionreader.files.append(parent+"/"+filename)
        return Actionreader.files

    def get_batch(self,batchsize,n_sequence,filenames, FileNumToPred = 17):
        if Actionreader.fileCount >= FileNumToPred:
          return None,None
        if not Actionreader.fo:
            print filenames[Actionreader.fileCount]
            print Actionreader.fileCount
            Actionreader.fo = open(filenames[Actionreader.fileCount], Actionreader.ACCESSMODE)
            line = Actionreader.fo.readline()
            #print "line=" + line
            j = 0
            nBatch_x = []
            while j < batchsize + 1 and line:
                k = 0
                oneSequence = []
                while k < n_sequence and line:
                    vector = line.split()
                    for i in range(len(vector)):
                        a = float(vector[i])
                        vector[i] = a
                    vector = vector[0:120]
                    oneSequence.append(vector[2:119])
                    k += 1
                    line = Actionreader.fo.readline()
                nBatch_x.append(oneSequence)
                j += 1
            nBatch_y = nBatch_x[1:]
            nBatch_x = nBatch_x[:batchsize]
            if not line and Actionreader.fileCount>=FileNumToPred:
                print Actionreader.fileCount
                print " >17"
                return None,None
            if not line and open(filenames[Actionreader.fileCount + 1], Actionreader.ACCESSMODE):
                Actionreader.fo.close()
                Actionreader.fo = None
                Actionreader.fileCount += 1
                return self.get_batch(batchsize, n_sequence, filenames)
            if not line and not open(filenames[Actionreader.fileCount + 1], Actionreader.ACCESSMODE):
                return None,None
            return nBatch_x,nBatch_y
        else:
            line = Actionreader.fo.readline()
            j=0
            nBatch_x=[]
            while j<batchsize+1 and line:
                k=0
                oneSequence = []
                while k<n_sequence and line:
                    vector=line.split()
                    for i in range(len(vector)):
                        a = float(vector[i])
                        vector[i] = a
                    vector = vector[0:120]
                    oneSequence.append(vector[2:119])
                    k+=1
                    line=Actionreader.fo.readline()
                nBatch_x.append(oneSequence)
                j+=1
            nBatch_y = nBatch_x[1:]
            nBatch_x = nBatch_x[:batchsize]
            if not line and Actionreader.fileCount>=FileNumToPred:
                print Actionreader.fileCount
                print " >17"
                return None,None
            if not line and open(filenames[Actionreader.fileCount+1], Actionreader.ACCESSMODE):
                Actionreader.fo.close()
                Actionreader.fo = None
                Actionreader.fileCount+=1
                return self.get_batch(batchsize,n_sequence,filenames)
            if not line and not open(filenames[Actionreader.fileCount+1], Actionreader.ACCESSMODE):
                return None,None
            return nBatch_x,nBatch_y

    def get_batch_FourNum(self,batchsize,n_sequence,filenames, FileNumToPred = 17,prelong = 1):
        if Actionreader.fileCount >= FileNumToPred:
          return None,None
        if not Actionreader.fo:
            print filenames[Actionreader.fileCount]
            print Actionreader.fileCount
            Actionreader.fo = open(filenames[Actionreader.fileCount], Actionreader.ACCESSMODE)
            line = Actionreader.fo.readline()
            #print "line=" + line
            j = 0
            nBatch_x = []
            while j < batchsize + prelong and line:
                k = 0
                oneSequence = []
                while k < n_sequence and line:
                    vector = line.split()
                    for i in range(len(vector)):
                        a = float(vector[i])
                        vector[i] = a
                    v = []
                    for o in range(9):
                        v.append(vector[o*13+11:o*13+15])
                    oneSequence.append(v)
                    k += 1
                    line = Actionreader.fo.readline()
                nBatch_x.append(oneSequence)
                j += 1
            nBatch_y = nBatch_x[1:]
            nBatch_x = nBatch_x[:batchsize]
            if not line and Actionreader.fileCount>=FileNumToPred:
                print Actionreader.fileCount
                print " >17"
                return None,None
            if not line and open(filenames[Actionreader.fileCount + 1], Actionreader.ACCESSMODE):
                Actionreader.fo.close()
                Actionreader.fo = None
                Actionreader.fileCount += 1
                return self.get_batch_FourNum(batchsize, n_sequence, filenames)
            if not line and not open(filenames[Actionreader.fileCount + 1], Actionreader.ACCESSMODE):
                return None,None
            return nBatch_x,nBatch_y
        else:
            line = Actionreader.fo.readline()

            j=0
            nBatch_x=[]
            while j<batchsize+prelong and line:
                k=0
                oneSequence = []
                while k<n_sequence and line:
                    vector=line.split()
                    for i in range(len(vector)):
                        a = float(vector[i])
                        vector[i] = a
                    v = []
                    for o in range(9):
                        v.append(vector[o * 13 + 11:o * 13 + 15])
                    oneSequence.append(v)
                    k+=1
                    line=Actionreader.fo.readline()
                nBatch_x.append(oneSequence)
                j+=1
            nBatch_y = nBatch_x[1:]
            nBatch_x = nBatch_x[:batchsize]
            if not line and Actionreader.fileCount>=FileNumToPred:
                print Actionreader.fileCount
                print " >17"
                return None,None
            if not line and open(filenames[Actionreader.fileCount+1], Actionreader.ACCESSMODE):
                Actionreader.fo.close()
                Actionreader.fo = None
                Actionreader.fileCount+=1
                return self.get_batch_FourNum(batchsize,n_sequence,filenames)
            if not line and not open(filenames[Actionreader.fileCount+1], Actionreader.ACCESSMODE):
                return None,None
            return nBatch_x,nBatch_y


    def process(self):
        fo=open(Actionreader.FileName,Actionreader.ACCESSMODE)
        line = fo.readline()
        while  line:
            vector=line.split()
            for i in vector:
                i = float(i)
            vector=vector[0:120]

            Actionreader.ClassLable.append(vector[119])
            Actionreader.FileVECTOR.append(vector[2:119])
            line = fo.readline()
        fo.close()
    def classprocess(self):
        for i in Actionreader.ClassLable:
            i=int(i)
            a= np.zeros(34,dtype=int)
            a[i]=1
            Actionreader.classlable.append(a)

    def out_data(self,pred,dir):
        fo = open(dir+"/pred-1", Actionreader.ACCESSMODEW)
        for sequence in pred[0]:
            fo.write(str(sequence))
            fo.write('\t')
        print pred
        fo.write('\n')
        fo.close()


r=Actionreader()
print



