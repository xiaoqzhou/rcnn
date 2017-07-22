import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from ReLU import ReLU

drop_p = 0.1
random_seed = 1234
from theano.tensor.shared_randomstreams import RandomStreams
rng = numpy.random.RandomState(random_seed)
def _dropout(rng, ivec, p):
  srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
  mask = srng.binomial(n=1, p=1-p, size=ivec.shape)
  output = ivec * T.cast(mask, theano.config.floatX)
  return output

class BidirectLstm(object):
   """"
   Bidirectional-RNN (recurrent neural network)
   """ 
	def __init__(self, input, params=None):
		self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.c0, \
		self.Wb_xi, self.Wb_hi, self.Wb_ci, self.bb_i, self.Wb_xf, self.Wb_hf, self.Wb_cf, self.bb_f, self.Wb_xc, self.Wb_hc, self.bb_c, self.Wb_xo, self.Wb_ho, self.Wb_co, self.bb_o, self.c1 = params    #load the parameters
		self.h0 = T.tanh(self.c0)
		self.h1 = T.tanh(self.c1)
		sigma = lambda x: 1 / (1 + T.exp(-x))
		def forward_step(x_t, h_tm1, c_tm1):          # forword RNN
			i_t = sigma(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
			f_t = sigma(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
			c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c)
			o_t = sigma(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co)  + self.b_o)
			h_t = o_t * T.tanh(c_t)
			return [h_t,c_t]
		[self.forward,self.f_context],_ = theano.scan(forward_step,sequences = input,outputs_info = [self.h0,self.c0])
		
		def backward_step(x_t,h_tm1,c_tm1):          # backword RNN
			i_t = sigma(T.dot(x_t, self.Wb_xi) + T.dot(h_tm1, self.Wb_hi) + T.dot(c_tm1, self.Wb_ci) + self.bb_i)
			f_t = sigma(T.dot(x_t, self.Wb_xf) + T.dot(h_tm1, self.Wb_hf) + T.dot(c_tm1, self.Wb_cf) + self.bb_f)
			c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.Wb_xc) + T.dot(h_tm1, self.Wb_hc) + self.bb_c)
			o_t = sigma(T.dot(x_t, self.Wb_xo)+ T.dot(h_tm1, self.Wb_ho) + T.dot(c_t, self.Wb_co) + self.bb_o)
			h_t = o_t * T.tanh(c_t)
			return [h_t,c_t]
		[self.tempward,self.b_context],_ = theano.scan(backward_step,sequences = input,outputs_info = [self.h1,self.c1],go_backwards = True)
		
		def check_step(x_t):
			o_t = x_t
			return o_t
		self.backward,_ =  theano.scan(check_step,sequences = self.tempward,outputs_info = None,go_backwards = True)   
		
		self.output = T.concatenate([self.forward,self.backward],axis=1)
		self.output = _dropout(rng, self.output, 0.1)
		self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.c0,\
					self.Wb_xi, self.Wb_hi, self.Wb_ci, self.bb_i, self.Wb_xf, self.Wb_hf, self.Wb_cf, self.bb_f, self.Wb_xc, self.Wb_hc, self.bb_c, self.Wb_xo, self.Wb_ho, self.Wb_co, self.bb_o, self.c1]
