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

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy 
import numpy as np

import os
import sys

default_path = '../'
path = os.path.abspath(os.path.join(os.path.dirname(__file__), default_path+'util/'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
import load_data as ld
from corrupt_input import *


class AutoEncoder(object):

    def __init__(self, numpy_rng, input=None, n_visible=784, n_hidden=500,
           W=None, R=None, bhid=None, bvis=None, symF=True, activation='sig'):


        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.numpy_rng = numpy_rng
        self.symF = symF
        self.activation = activation
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W')

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                        dtype=theano.config.floatX), name='bvis')

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                              dtype=theano.config.floatX), name='bhid')
        
        if not symF and not R:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            R = theano.shared(value=initial_W.T, name='R')

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis

        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.fmatrix(name='input')
        else:
            self.x = input

        # tied weights, therefore W_prime is W transpose
        if symF:
            self.W_prime = self.W.T 
            self.params = [self.W, self.b, self.b_prime] 
        else:
            self.W_prime = R
            self.params = [self.W, self.W_prime, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """

        if self.activation == 'relu':
            z = T.dot(input, self.W) + self.b
            return T.switch(z<0, 0, z)
        elif self.activation == 'tanh':
            return T.tanh(T.dot(input, self.W) + self.b)
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    
    def get_reconstructed_input(self, hidden, binary=False):
        """ Computes the reconstructed input given the values of the hidden layer """
        if binary:
            return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        else:
            return T.dot(hidden, self.W_prime) + self.b_prime

    def get_reconstructed_input_given_x(self,input, binary=False):
        hidden = self.get_hidden_values(input)
        recon_x= self.get_reconstructed_input(hidden, binary=binary)
        return recon_x

    def get_reconstructed_err(self, input, binary=False):
        """ Computes the reconstructed input given the values of the input layer """
        hidden = self.get_hidden_values(input)
        recon_x= self.get_reconstructed_input(hidden, binary=binary)
        if binary:
            return T.mean( - T.sum(X * T.log(recondX) + (1-X) * T.log(1 - recondX), axis=1))
        else:
            return T.mean(T.sum(0.5*(input - recon_x)*(input-recon_x),axis=1))

    def weight_decay(self):
        return (self.W ** 2).sum() /2

    def divFreeEnergy(self, X, batch_sz):
        hidden  = self.get_hidden_values(X)
        J = T.reshape(hidden * (1 - hidden), (batch_sz, 1, self.n_hidden))\
                    * T.reshape(self.W**2, (1, self.n_visible, self.n_hidden))
        return T.sum(J) / batch_sz

    def contractive_pen(self, X, batch_sz):

        y = self.get_hidden_values(X)
        J = self.get_jacobian(y, self.W, batch_sz)
        # Compute the jacobian and average over the number of samples/minibatch
        return T.sum(J ** 2) / batch_sz
 

    def get_jacobian(self, hidden, W, batch_sz):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        """
        return T.reshape(hidden * (1 - hidden),
                         (batch_sz, 1, self.n_hidden)) * T.reshape(
                             W.T, (1, self.n_visible, self.n_hidden))

    def cost(self, X, corrupt_in=0, binary=True, ntype='zeromask'):
        """ This function computes the cost and the updates for one trainng
        step """
        #Corrupt input 
        corrupt_x = get_corrupted_input(self.numpy_rng, X, corrupt_in, ntype=ntype)

        y = self.get_hidden_values(corrupt_x)
        recondX = self.get_reconstructed_input(y, binary)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        if binary:
            L = - T.sum(X * T.log(recondX) + (1-X) * T.log(1 - recondX), axis=1)
        else:
            L = T.sum(0.5*(recondX-X)*(recondX-X), axis=1)

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        return cost

if __name__ == '__main__':
    autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(123), n_visible=784, n_hidden=500)
    cost, updates = autoencoder.get_cost_updates(learning_rate=0.1)
    train = theano.function([x], cost, updates=updates)

