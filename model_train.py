import cPickle as pickle
import gzip
import os
import sys
import time
import numpy
import math
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import prepare 
import parameters
from LeNetConvPoolLayer import LeNetConvPoolLayer
from mlp import HiddenLayer
from BidirectLstm import BidirectLstm
from ReLU import ReLU 
from theano.tensor.shared_randomstreams import RandomStreams
theano.config.openmp=True

"""
  Loading Hyper-parameters
"""
drop_p = parameters.drop_p
random_seed = 1234
sent_len = 100
dimen = int(math.sqrt(sent_len))
word_len = 3*parameters.Word_Len
C_num = parameters.C_NO
rng = numpy.random.RandomState(random_seed)
learning_rate = parameters.learning_rate
iternum = parameters.Internum
nkerns = parameters.nkerns
filter_shape0 = parameters.filter_shape0
filter_shape1 = parameters.filter_shape1
filter_shape2 = parameters.filter_shape2
image_shape0 = parameters.image_shape0
image_shape1 = parameters.image_shape1
image_shape2 = parameters.image_shape2
batch_size = parameters.batch_size

startpara = 0
i = 1
while(i < len(sys.argv)):
  arg = sys.argv[i]
  if arg == '-train':
    startpara = float(sys.argv[i+1])
  else:
    pass
  i = i + 2
 
def _dropout_from_layer(rng, layer, p):  # The function of Dropout in architecture
  srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
  mask = srng.binomial(n=1, p=1-p, size=layer.shape)
  output = layer * T.cast(mask, theano.config.floatX)
  return output
###################################################################################################################
class RCNN_Match(object):
    """"
    The Architecture of RCNN
    """
	def __init__(self, P_input1, P_input2, P_y,params=None):
		c_w0, c_b0, c_w1, c_b1, c_w2, c_b2, c_w3, c_b3, c_w4, c_b4, c_w5, c_b5, m_w1, m_b1,\
		W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, c0, \
		Wb_xi, Wb_hi, Wb_ci, bb_i, Wb_xf, Wb_hf, Wb_cf, bb_f, Wb_xc, Wb_hc, bb_c, Wb_xo, Wb_ho, Wb_co, bb_o, c1, W_hy, b_y = params
		rnnparams0 = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, c0, Wb_xi, Wb_hi, Wb_ci, bb_i, Wb_xf, Wb_hf, Wb_cf, bb_f, Wb_xc, Wb_hc, bb_c, Wb_xo, Wb_ho, Wb_co, bb_o, c1]
		def cnn_step(q_t, a_t):
			q_matrix = theano.tensor.reshape(q_t, [sent_len,word_len])
			a_matrix = theano.tensor.reshape(a_t, [sent_len,word_len])
			q_tensor3 = theano.tensor.reshape(q_matrix, [dimen,dimen,word_len])
			a_tensor3 = theano.tensor.reshape(a_matrix, [dimen,dimen,word_len])
			q_input = q_tensor3.dimshuffle('x', 2, 1, 0)
			a_input = a_tensor3.dimshuffle('x', 2, 1, 0)   # QA -> Tensor4
			n_layer0 = LeNetConvPoolLayer(q_input, filter_shape=filter_shape0, image_shape=image_shape0,\
										W=c_w0, b=c_b0, poolsize=(2, 2))
			n_input1 = n_layer0.output
			n_layer1 = LeNetConvPoolLayer(n_input1, filter_shape=filter_shape1, image_shape=image_shape1,\
										W=c_w1, b=c_b1, poolsize=(2, 2))
			n_input2 = n_layer1.output
			n_layer2 = LeNetConvPoolLayer(n_input2, filter_shape=filter_shape2, image_shape=image_shape2,\
										W=c_w2, b=c_b2, poolsize=(1, 1))

			n_layer3 = LeNetConvPoolLayer(a_input, filter_shape=filter_shape0, image_shape=image_shape0,\
										W=c_w3, b=c_b3, poolsize=(2, 2))
			n_input4 = n_layer3.output
			n_layer4 = LeNetConvPoolLayer(n_input4, filter_shape=filter_shape1, image_shape=image_shape1,\
										W=c_w4, b=c_b4, poolsize=(2, 2))
			n_input5 = n_layer4.output
			n_layer5 = LeNetConvPoolLayer(n_input5, filter_shape=filter_shape2, image_shape=image_shape2,\
										W=c_w5, b=c_b5, poolsize=(1, 1))
			layer2_input1 = n_layer2.output.flatten(2)
			layer2_input2 = n_layer5.output.flatten(2)
			x_t = T.concatenate([layer2_input1,layer2_input2],axis=1).flatten(1)
			return x_t
		self.cnn_output,_ = theano.scan(cnn_step,sequences = [P_input1,P_input2],outputs_info = None)
		self.cnn_output = _dropout_from_layer(rng, self.cnn_output, drop_p[0])                     # Above is The module of CNN-QA-matching 
		##########################################################################
		self.hidden_input = self.cnn_output
		hidden_layer = HiddenLayer(self.hidden_input, W=m_w1, b=m_b1)
		self.hidden_out = _dropout_from_layer(rng, hidden_layer.output, drop_p[4])
		Brnn_layer0 = BidirectLstm(self.hidden_out,params = rnnparams0)                           # Here is The module of Answer sequence learning 
		##########################################################################
		self.output = Brnn_layer0.output
		self.y_pred = T.dot(self.output, W_hy) + b_y
		
		self.params = params
		self.p_y_given_x = T.nnet.softmax(self.y_pred)
		self.result = T.argmax(self.p_y_given_x, axis=-1)
		self.cost =  -T.mean(T.log(self.p_y_given_x)[T.arange(P_y.shape[0]), P_y])
		if P_y.dtype.startswith('int'):
			self.error = T.mean(T.neq(self.result, P_y))

class train_model(object):
  """"
   The optimization of of RCNN
  """
  def __init__(self,learning_rate,modelparams):
    self.params = modelparams
    self.learning_rate = learning_rate

  def training(self,P_input1,P_input2,P_y):   # The optimization of model parameters (Adagrad method) 
    self.rcnn_match = RCNN_Match(P_input1,P_input2,P_y,params = self.params)
    self.out = self.rcnn_match.result
    self.cost = self.rcnn_match.cost
    updates = []
    m_accumulator = []
    for para_i in self.params:
		eps_p = numpy.zeros_like(para_i.get_value(borrow=True), dtype=theano.config.floatX)
		m_accumulator.append(theano.shared(eps_p, borrow=True))
    self.gparams = T.grad(self.cost,self.params)
    for param_i, grad_i, acc_i in zip(self.params, self.gparams, m_accumulator):
		acc = acc_i + T.sqr(grad_i)
		updates.append((param_i, param_i - self.learning_rate * grad_i / (T.sqrt(acc) + 1e-5)))
		updates.append((acc_i, acc))
    self.updates = updates
    return self.out,self.cost

  def storedata(self,index,wpath):  # Save the model after each epoch 
    ofile=open(wpath + str(index)+"stack_paras.pkl",'wb')
    pickle.dump(self.params, ofile)
    ofile.close()

def loadparameters(filename):
  ifile = open(filename,'rb')
  params = pickle.load(ifile)
  ifile.close()
  return params
  
  
########################################
""""
 The main Function of model learning
"""
#######################################
def train_worker(ipath,wpath):     
  params = parameters.random_weights()
  P_input1 = T.matrix(name='P_input1',dtype=theano.config.floatX)
  P_input2 = T.matrix(name='P_input2',dtype=theano.config.floatX)
  P_y = T.vector(name='P_y',dtype='int32')
  RCNN = train_model(learning_rate,params)
  p_y,cost = RCNN.training(P_input1,P_input2,P_y)
  rcnn = theano.function(inputs=[P_input1,P_input2,P_y],outputs = [p_y,cost],updates=RCNN.updates)
  print 'reading dataset'
  prep = prepare.prepareData(ipath)
  prep.Reading_traindata()
  block_size = prep.block_size
  session_size = prep.train_session_num

  for epoch in range(iternum):
	print "epoch", epoch
	for k in range(session_size):
		item = prep.generate_batch_from_sentence(k)
		out,cost = rcnn(item[0],item[1],item[2])
		sys.stdout.flush()
		sys.stdout.write(str(k)+'\r')
	RCNN.storedata(epoch,wpath)

if __name__ == '__main__':
    start = time.time()
    iPath = './data/'  
    wPath = './WeightModel/'
    train_worker(iPath,wPath)
    print >> sys.stderr, "ok"
    print >> sys.stderr, "all toke", float(time.time()-start)/60.,"min"
