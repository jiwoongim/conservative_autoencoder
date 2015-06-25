''' Version 1.000
 Code provided by Daniel Jiwoong Im and Mohamed Ishmael Belghazi

 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

'''Demo of Conservativeness of Autoencoders.
For more information, see :http://arxiv.org/...
'''

import numpy as np
import pylab
import theano
import theano.tensor as T


'''Initialize the weight of neural network'''
def initialize_weight(n_vis, n_hid, W_name, numpy_rng, rng_dist):

    if 'uniform' in rng_dist:
        W = numpy_rng.uniform(low=-np.sqrt(6. / (n_vis + n_hid)),\
                high=np.sqrt(6. / (n_vis + n_hid)),
                size=(n_vis, n_hid)).astype(theano.config.floatX)
        #if 'exp' in rng_dist :
        #    W = np.exp(-W)
    elif rng_dist == 'normal':
        W = 0.01 * numpy_rng.normal(size=(n_vis, n_hid)).astype(theano.config.floatX)

    return theano.shared(value = W, name=W_name)

'''Initialize the bias'''
def initialize_bias(n, b_name):

    return theano.shared(value = np.zeros((n,), \
                dtype=theano.config.floatX), name=b_name)


'''decaying learning rate'''
def get_epsilon(epsilon, n, i):
    return epsilon / ( 1 + i/float(n))


def dist2hy(x,y):
    '''Distance matrix computation
    Hybrid of the two, switches based on dimensionality
    '''
    #if x.eval().shape[1]<5:  #If the dimension is small
    #    d = T.zeros_like(T.dot(x, y.T))
    #    #d = np.zeros((x.shape[0],y.shape[0]),dtype=x.dtype)
    #    for i in xrange(x.eval().shape[1]):
    #        diff2 = x[:,i,None] - y[:,i]
    #        diff2 **= 2
    #        d += diff2
    #    #np.sqrt(d,d)
    #    return d

    #else:
    d = T.dot(x,y.T)
    d *= -2.0
    d += T.sum(x*x, axis=1).dimshuffle(0,'x')
    d += T.sum(y*y, axis=1)
    # Rounding errors occasionally cause negative entries in d
    d = d * T.cast(d>0,theano.config.floatX)
    #d[d<0] = 0
    # in place sqrt
    #np.sqrt(d,d)
    return T.sqrt(d)


'''Computes Euclidean distance in numpy'''
def euclidean_np(A,B):
    n = A.shape[0]
    m = B.shape[0]
    AA = np.sum((A * A),axis=1)
    BB = np.sum((B * B),axis=1)
        
    dist = np.tile(AA,(n,1)).T +np.tile(BB,(m,1)) - 2 * np.dot(A,B.T)
    return np.sqrt(dist)


'''Rectified Linear unit'''
def Relu(x):
    return T.maximum(0,x)



