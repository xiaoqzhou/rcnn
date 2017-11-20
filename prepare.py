import numpy
import math
import parameters
import theano
import theano.tensor as T
import scipy.spatial
import random
import time
import cPickle as pickle
'''
 this is the load data file
'''
Word_Len = parameters.Word_Len
class prepareData(object):
  """"
  Data preprocessing
  """
  def __init__(self,inputpath):
    para = parameters.Parameters()
    para.readEmbeddeing()
    self.embeddings = para.getEmbedding()
    self.word2id = para.getWord2id()
    self.train_file = inputpath + 'task3_4label_train.txt'
    self.test_file = inputpath +  'task3_4label_devel.txt'
    self.train_list = inputpath + 'task3_4label_train.list'
    self.test_list = inputpath + 'task3_4label_devel.list'
    ##################################
    self.sen_len = 100 # the max len of sentence
    self.rng = random.Random()
    self.rng.seed(2014)
    self.train_sentence = []
    self.test_sentence = []
    self.sentence = []
    self.train_index = [] # the train index
    self.test_index = []  # the test index
    self.neg = 1
     
    self.tmp1 = numpy.zeros([3*Word_Len],dtype = float)
    self.tmp2 = numpy.zeros([3*Word_Len],dtype = float)
    self.allZero = numpy.zeros([Word_Len],dtype = float)
    self.block_size = 1
	
  ################################################################################
  def Reading_traindata(self):   # Loading the training QA data
    j = 0
    with open(self.train_file,'r') as fin:
      for line in fin:
        subsentence = line.split('\t')
        self.train_sentence.append((subsentence[0], subsentence[1].lower(), subsentence[2].lower()))
        j = j + 1

    train_session_list = {};
    train_dialog  = []
    l = 0
    with open(self.train_list,'r') as fin:
	  for line in fin:
		index = line.split('\t')
		self.train_index.append((index[0],index[1],index[2]))
		pos = index[1].find('Q')+1
		qid = int(index[1][pos:len(index[1])])

		if(train_session_list.has_key(index[0])==False):
			train_session_list[index[0]] = []
			train_session_list[index[0]].append(l)
			a_vec = []
			a_vec.append(l)
			train_dialog.append(a_vec)
		else:
			train_session_list[index[0]].append(l)
			train_dialog[qid-1].append(l)
		l = l + 1
    
    for i in range(len(self.train_sentence)):
		self.sentence.append(self.train_sentence[i])
    train_len = len(self.sentence)

    self.train_session = train_dialog
    print 'train dialog is ',len(self.train_session)
    self.train_session_num = len(self.train_session)
    self.train_num = train_len

  def Reading_testdata(self):  # Loading the test QA data
    j = 0
    with open(self.test_file,'r') as fin:
      for line in fin:
        subsentence = line.split('\t')
        self.test_sentence.append((subsentence[0], subsentence[1].lower(), subsentence[2].lower()))
        j = j + 1
    test_len = len(self.test_sentence)
    test_session_list = {}
    test_dialog = [] 
  
    distance = 0
    l = 0
    qid = 0
    with open(self.test_list,'r') as fin:
      for line in fin:
		index = line.split('\t')
		self.test_index.append((index[0],index[1],index[2]))
		if(test_session_list.has_key(index[0])==False):
			test_session_list[index[0]] = []
			test_session_list[index[0]].append(l)
			a_vec = [] 
			a_vec.append(l)
			test_dialog.append(a_vec)
			qid = qid + 1
		else:
			test_session_list[index[0]].append(l)
			test_dialog[qid-1].append(l)
		l = l + 1
    self.test_session = test_dialog
    self.test_session_num = len(self.test_session)
    self.test_num = test_len

  def fromtext2vector(self,subsentence1, subsentence2): # Convert QA pair into the inputting tensor4 of RCNN
    qsentence_match = numpy.asarray(numpy.zeros([self.sen_len,3*Word_Len],dtype='float32'),dtype ='float32') #
    asentence_match = numpy.asarray(numpy.zeros([self.sen_len,3*Word_Len],dtype='float32'),dtype ='float32')

    sentence1 = subsentence1.split()
    sentence2 = subsentence2.split()
    dimen = int(math.sqrt(self.sen_len))
    cnt = 0
    for k in range(len(sentence1)):
      try:
        index = self.word2id[sentence1[k]]
      except KeyError:
        index = self.word2id['unknown']
      self.tmp1 = self.embeddings[index]  
      if (k+1) > (len(sentence1)-1):
        self.tmp1 = numpy.concatenate((self.tmp1,self.allZero))  
        self.tmp1 = numpy.concatenate((self.tmp1,self.allZero))  
      elif (k+2) > (len(sentence1)-1):
        try:
          index = self.word2id[sentence1[k+1]]
        except KeyError:
          index = self.word2id['unknown']
        self.tmp1 = numpy.concatenate((self.tmp1,self.embeddings[index]))  
        self.tmp1 = numpy.concatenate((self.tmp1,self.allZero))            
      else:
        try:
          index = self.word2id[sentence1[k+1]]
        except KeyError:
          index = self.word2id['unknown']
        self.tmp1 = numpy.concatenate((self.tmp1,self.embeddings[index])) 
        try:
          index = self.word2id[sentence1[k+2]]
        except KeyError:
          index = self.word2id['unknown']
        self.tmp1 = numpy.concatenate((self.tmp1,self.embeddings[index]))  
      qsentence_match[cnt] = self.tmp1
      cnt = cnt + 1
    qsentence_match =  qsentence_match.flatten()

    cnt = 0
    for l in range(len(sentence2)):
      try:
        index = self.word2id[sentence2[l]]
      except KeyError:
        index = self.word2id['unknown']
      self.tmp2 = self.embeddings[index]
      if (l+1) > (len(sentence2)-1):
        self.tmp2 = numpy.concatenate((self.tmp2,self.allZero))
        self.tmp2 = numpy.concatenate((self.tmp2,self.allZero))
      elif (l+2) > (len(sentence2)-1):
        try:
          index = self.word2id[sentence2[l+1]]
        except KeyError:
          index = self.word2id['unknown']
        self.tmp2 = numpy.concatenate((self.tmp2,self.embeddings[index]))
        self.tmp2 = numpy.concatenate((self.tmp2,self.allZero))
      else:
        try:
          index = self.word2id[sentence2[l+1]]
        except KeyError:
          index = self.word2id['unknown']
        self.tmp2 = numpy.concatenate((self.tmp2,self.embeddings[index]))
        try:
          index = self.word2id[sentence2[l+2]]
        except KeyError:
          index = self.word2id['unknown']
        self.tmp2 = numpy.concatenate((self.tmp2,self.embeddings[index])) 
      asentence_match[cnt] = self.tmp2
      cnt = cnt + 1
    asentence_match =  asentence_match.flatten()
    return qsentence_match,asentence_match
    
  def generate_batch_from_sentence(self,session_id):  # Getting one training sample : [Q,A1,A2,...AN]
    index = 0
    session = self.train_session[session_id]
    batch_size = len(session)
    dimen = int(math.sqrt(self.sen_len))
    qbatch = numpy.asarray(numpy.zeros([batch_size,self.sen_len*3*Word_Len],dtype='float32'),dtype ='float32')
    abatch = numpy.asarray(numpy.zeros([batch_size,self.sen_len*3*Word_Len],dtype='float32'),dtype = 'float32')
    label = numpy.asarray(numpy.zeros(batch_size,dtype='int32'),dtype='int32')
    for i in range(len(session)):
		pid = session[i]
		qbatch[index],abatch[index] = self.fromtext2vector(self.sentence[pid][1],self.sentence[pid][2])
		tag  = int(self.sentence[pid][0])
		if tag==3:
			label[index] = 2
		else:
			label[index] = tag
		index = index + 1
    return [qbatch,abatch,label]
  
  def generate_test_from_sentence(self,session_id): # Getting one test sample : [Q,A1,A2,...AN]
    index = 0
    session = self.test_session[session_id]
    batch_size = len(session)
    dimen = int(math.sqrt(self.sen_len))
    qbatch = numpy.asarray(numpy.zeros([batch_size,self.sen_len*3*Word_Len],dtype='float32'),dtype ='float32')
    abatch = numpy.asarray(numpy.zeros([batch_size,self.sen_len*3*Word_Len],dtype='float32'),dtype = 'float32')	
    label = numpy.asarray(numpy.zeros([batch_size,], dtype='int32'),dtype='int32')
    for i in range(len(session)):
		pid = session[i]
		qbatch[index],abatch[index] = self.fromtext2vector(self.test_sentence[pid][1],self.test_sentence[pid][2])
		tag  = int(self.test_sentence[pid][0])
		if tag==3:
			label[index] = 2
		else:
			label[index] = tag
		index = index + 1
    return [qbatch,abatch,label]
  

if __name__ =="__main__":
  ipath= './data/'
  start = time.time()
  prep = prepareData(ipath)
  prep.Reading_testdata()
  print 'beautiful!','it takes',time.time()-start
