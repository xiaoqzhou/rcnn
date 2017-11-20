__author__ = 'Xiaoqiang Zhou'

import cPickle as pickle
import math
import numpy
import theano

floatX = theano.config.floatX

drop_p = [0.1,0.1,0.1,0.1,0.1]               # The dropout setup for each layers

Word_Len = 100
learning_rate = 0.01
batch_size = 1
Internum = 25                                #The number of epoches in training

nkerns = [100,100,100]                       # The number of feature maps in each convolutiaon layers
filter_shape0 = (nkerns[0],3*Word_Len,1,1)   
filter_shape1 = (nkerns[1],nkerns[0],2,2) 
filter_shape2 = (nkerns[2],nkerns[1],2,2)  

image_shape0 = (batch_size, 3*Word_Len,10,10)
image_shape1 = (batch_size, nkerns[0],5,5) 
image_shape2 = (batch_size, nkerns[1],2,2)
#Here is the setup of the hyper-parameters for CNN

hidden_in  = nkerns[2]*2
hidden_out = 480
Class_NO = 3                  # The number of tagert classes

lstm_hidden = 480
n_i = hidden_out
n_h = lstm_hidden
n_o = lstm_hidden
n_y = Class_NO
#Here is the setup of the hyper-parameters for RNN


def rnn_weights(isize,hsize,osize):
    """"
    The parameters in one Bidirectional-RNN
    """
	rng = numpy.random.RandomState(2014)
	i = isize
	h = hsize
	o = osize
	W_xi = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	W_hi = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	W_ci = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	b_i = numpy.zeros((h,), dtype=floatX)
	W_xf = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	W_hf = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	W_cf = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	b_f = numpy.zeros((h,), dtype=floatX)
	W_xc = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	W_hc = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	b_c =  numpy.zeros((h,), dtype=floatX)
	W_xo = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	W_ho = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	W_co = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	b_o = numpy.zeros((h,), dtype=floatX)
	c0 = numpy.zeros((h,), dtype=floatX)
	
	Wb_xi = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	Wb_hi = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	Wb_ci = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	bb_i = numpy.zeros((h,), dtype=floatX)
	Wb_xf = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	Wb_hf = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	Wb_cf = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	bb_f = numpy.zeros((h,), dtype=floatX)
	Wb_xc = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	Wb_hc = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	bb_c =  numpy.zeros((h,), dtype=floatX)
	Wb_xo = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(i+h)),high=numpy.sqrt(6./(i+h)),size=(i,h)),dtype=floatX)
	Wb_ho = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	Wb_co = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(h+h)),high=numpy.sqrt(6./(h+h)),size=(h,h)),dtype=floatX)
	bb_o = numpy.zeros((h,), dtype=floatX)
	c1 = numpy.zeros((h,), dtype=floatX)
	return [theano.shared(W_xi, borrow = True),theano.shared(W_hi, borrow = True),\
           	theano.shared(W_ci, borrow = True),theano.shared(b_i, borrow = True),\
           	theano.shared(W_xf, borrow = True),theano.shared(W_hf, borrow = True),\
           	theano.shared(W_cf, borrow = True),theano.shared(b_f, borrow = True),\
           	theano.shared(W_xc, borrow = True),theano.shared(W_hc, borrow = True),\
           	theano.shared(b_c, borrow = True),theano.shared(W_xo, borrow = True),\
           	theano.shared(W_ho, borrow = True),theano.shared(W_co, borrow = True),\
           	theano.shared(b_o, borrow = True),theano.shared(c0, borrow = True),\
           	theano.shared(Wb_xi, borrow = True),theano.shared(Wb_hi, borrow = True),\
           	theano.shared(Wb_ci, borrow = True),theano.shared(bb_i, borrow = True),\
           	theano.shared(Wb_xf, borrow = True),theano.shared(Wb_hf, borrow = True),\
           	theano.shared(Wb_cf, borrow = True),theano.shared(bb_f, borrow = True),\
           	theano.shared(Wb_xc, borrow = True),theano.shared(Wb_hc, borrow = True),\
           	theano.shared(bb_c, borrow = True),theano.shared(Wb_xo, borrow = True),\
           	theano.shared(Wb_ho, borrow = True),theano.shared(Wb_co, borrow = True),\
           	theano.shared(bb_o, borrow = True),theano.shared(c1, borrow = True)]

def random_weights():
    """"
    All the parameters of the whole model (RCNN)
    """
   	rng = numpy.random.RandomState(2014)
   	c_w0 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),\
                        	high=numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),size=filter_shape0),dtype=floatX) 
   	c_b0 = numpy.zeros((filter_shape0[0],), dtype=floatX)
   
   	c_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),\
                        	high=numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),size=filter_shape1),dtype=floatX)
   	c_b1 = numpy.zeros((filter_shape1[0],), dtype=floatX)

   	c_w2 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),\
                        	high=numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),size=filter_shape2),dtype=floatX)
   	c_b2 = numpy.zeros((filter_shape2[0],), dtype=floatX)

   	c_w3 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),\
                        	high=numpy.sqrt(6./(filter_shape0[0]+filter_shape0[1])),size=filter_shape0),dtype=floatX)
   	c_b3 = numpy.zeros((filter_shape0[0],), dtype=floatX)

   	c_w4 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),\
                        	high=numpy.sqrt(6./(filter_shape1[0]+filter_shape1[1])),size=filter_shape1),dtype=floatX)
   	c_b4 = numpy.zeros((filter_shape1[0],), dtype=floatX)

   	c_w5 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),\
                        	high=numpy.sqrt(6./(filter_shape2[0]+filter_shape2[1])),size=filter_shape2),dtype=floatX)
   	c_b5 = numpy.zeros((filter_shape2[0],), dtype=floatX)

   	m_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(hidden_in + hidden_out)),\
                        	high=numpy.sqrt(6./(hidden_in+hidden_out)),size=(hidden_in,hidden_out)),dtype=floatX)
   	m_b1 = numpy.zeros((hidden_out,), dtype=floatX)

   	cnnparams = [theano.shared(c_w0, borrow = True),theano.shared(c_b0, borrow = True),\
   				theano.shared(c_w1, borrow = True),theano.shared(c_b1, borrow = True),\
   				theano.shared(c_w2, borrow = True),theano.shared(c_b2, borrow = True),\
   				theano.shared(c_w3, borrow = True),theano.shared(c_b3, borrow = True),\
   				theano.shared(c_w4, borrow = True),theano.shared(c_b4, borrow = True),\
   				theano.shared(c_w5, borrow = True),theano.shared(c_b5, borrow = True),\
   				theano.shared(m_w1, borrow = True),theano.shared(m_b1, borrow = True)]                   # CNN parameters

   	rnnparams0 = rnn_weights(n_i,n_h,n_o)                                                                #RNN parameters
   	W_hy = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(2*n_o + n_y)),high=numpy.sqrt(6./(2*n_o+n_y)),size=(2*n_o,n_y)),dtype=floatX)
   	b_y = numpy.zeros((n_y,), dtype=floatX)
   	mlpparams = [theano.shared(W_hy, borrow = True),theano.shared(b_y, borrow = True)]           # The parameters of the top layer
   	modelparams = cnnparams + rnnparams0 + mlpparams
   	return modelparams

class Parameters:                             #The Class : Loading the pre-trained word2vec for representation the sentence
    """
    Word Embeddiing used by the Model
    """

    def __init__(self):
      self.embeddings_path = './wordEmbeddings/Task3_Embedding100.txt'
      self.word2id = {}
      self.words= [] 
    def readEmbeddeing(self):
      with open(self.embeddings_path,'r') as f:
        line = f.readline()
        vocab_size, embedding_size = line.strip('\n').split()
        self.vocab_size = int(vocab_size)
        self.embedding_size = int(embedding_size)

        self.embeddings = numpy.asarray(numpy.random.rand(self.vocab_size, self.embedding_size),dtype=float)
        self.embeddings = self.embeddings * 0
        for i in range(self.vocab_size):
          line = f.readline()
          tmp_embedding = line.strip('\n').split()
          self.words.append(tmp_embedding[0])
          self.word2id[tmp_embedding[0]] = i
          tmp_embedding = tmp_embedding[1:]
          tmp_embedding = [float(elem) for elem in tmp_embedding]
          self.embeddings[i] = tmp_embedding
 
    def getEmbedding(self):
      return self.embeddings        # return the matriex of word2vec

    def getWord2id(self):
      return self.word2id           # return the dictionary of word2vec
   
    def getParsembedding(self):
      return self.parsembeddings


if __name__ == "__main__":
  para = Parameters()
  para.readEmbeddeing()
  print para.word2id['good']
