#Copyright (c) 2013, Hanna Kamyshanska
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
#            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
#            BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#            IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#            OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#            OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#            OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###
"""
Contractive autoencoder:

Contractive regularizer: Frobenius norm of Jacobian Matrix

Implemented activation functions: 'sigmoid' (default);
    'relu' (half-way rectifier function);
    'softplus';
    'tanh';
    'quadratic';
    'linear'.

In case of real-valued inputs (BINARY = False) linear output activaation and squared error loss are used (default);
in case of binary inputs (BINARY = True) the autoencoder has sigmoid output activaation and cross-entropy loss is used

"""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import *

SMALL = 0.000001
class cAE(object):
    def __init__(self, n_visible, n_hiddens, contraction = 0.5, activation='sigmoid', n_samples = None, params=None, BINARY = False, MINIBATCH_SIZE =100, corruptF=False):
        self.n_visible = n_visible
        self.data_dim=np.sqrt(n_visible)
        self.n_hiddens =n_hiddens
        self.contraction =contraction
        self.activation = activation
        if BINARY:
            self.vistype= 'binary'
            print 'binary'
        else:
            self.vistype= 'real'

        numpy_rng  = np.random.RandomState(1)
        theano_rng = RandomStreams(1)

        ### Loar pretrained parameters or initialize parameters at random:
        if  params:
            [W_np, b_hid_np, b_out_np] = params
            self.W = theano.shared(value = W_np, name ='W')
            self.b_out = theano.shared(value=b_out_np, dtype=theano.config.floatX, name='b_out')
            self.b_hid = theano.shared(value=b_hid_np, dtype=theano.config.floatX, name='b_hid')
        else:
            self.W_init = np.asarray(numpy_rng.uniform(low  = -4*np.sqrt(6./(n_hiddens+n_visible)), high =  4*np.sqrt(6./(n_hiddens+n_visible)), size = (n_visible, n_hiddens)), dtype = theano.config.floatX)
            self.W = theano.shared(value = self.W_init, name ='W')
            self.b_out = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), name='bvis')
            self.b_hid = theano.shared(value=np.zeros(n_hiddens, dtype=theano.config.floatX), name ='bhid')

        if corruptF:
            self.inputs = get_corrupted_input(numpy_rng, T.matrix(name = 'inputs'), 0.01, ntype='gaussian')
        else:
            self.inputs = T.matrix(name = 'inputs')
        self.params = [self.W, self.b_hid, self.b_out]

        ### choose activation function
        if activation == 'sigmoid':
            self._hiddens = T.nnet.sigmoid(T.dot(self.inputs, self.W) + self.b_hid)
            contractive_cost =  T.sum( ((self._hiddens * (1 - self._hiddens))**2) * T.sum(self.W**2, axis=0), axis=1)
            if self.vistype == 'binary':
                self._outputs = T.nnet.sigmoid(T.dot(self._hiddens, self.W.T) + self.b_out)
            elif self.vistype == 'real':
                self._outputs = T.dot(self._hiddens, self.W.T) + self.b_out

        elif activation == 'relu':
            score = (T.dot(self.inputs, self.W) + self.b_hid)
            self._hiddens = (T.sgn(score)+1)*score*0.5
            contractive_cost =  T.sum( (T.sgn(score)+1)*0.5* T.sum(self.W**2, axis=0), axis=1)
            if self.vistype == 'binary':
                self._outputs = T.nnet.sigmoid(T.dot(self._hiddens, self.W.T) + self.b_out)
            elif self.vistype == 'real':
                self._outputs = T.dot(self._hiddens, self.W.T) + self.b_out

        elif activation == 'softplus':
            self._hiddens = T.nnet.softplus(T.dot(self.inputs, self.W) + self.b_hid)
            contractive_cost =  T.sum( T.nnet.sigmoid(T.dot(self.inputs, self.W) + self.b_hid) * T.sum(self.W**2, axis=0), axis=1)
            if self.vistype == 'binary':
                self._outputs = T.nnet.softplus(T.dot(self._hiddens, self.W.T) + self.b_out)
            elif self.vistype == 'real':
                self._outputs = T.dot(self._hiddens, self.W.T) + self.b_out

        elif activation == 'tanh':
            self._hiddens = T.tanh(T.dot(self.inputs, self.W) + self.b_hid)
            contractive_cost = T.sum( ( 1 - self._hiddens**2) * T.sum(self.W**2, axis=0), axis=1)
            if self.vistype == 'binary':
                self._outputs = T.tanh(T.dot(self._hiddens, self.W.T) + self.b_out)
            elif self.vistype == 'real':
                self._outputs = T.dot(self._hiddens, self.W.T) + self.b_out

        elif activation == 'quadratic':
            self._hiddens = (T.dot(self.inputs, self.W) + self.b_hid)*(T.dot(self.inputs, self.W) + self.b_hid)
            contractive_cost = 0# not implemented yet
            if self.vistype == 'binary':
                self._outputs = (T.dot(self._hiddens, self.W.T) + self.b_out)*(T.dot(self._hiddens, self.W.T) + self.b_out)
            elif self.vistype == 'real':
                self._outputs = T.dot(self._hiddens, self.W.T) + self.b_out

        elif activation == 'linear':
            self._hiddens = (T.dot(self.inputs, self.W) + self.b_hid)
            contractive_cost = T.sum(T.sum(self.W**2, axis=0))
            if self.vistype == 'binary':
                self._outputs = T.nnet.sigmoid((T.dot(self._hiddens, self.W.T) + self.b_out))
            elif self.vistype == 'real':
                self._outputs = T.dot(self._hiddens, self.W.T) + self.b_out
        else:
            print 'Not implemented activation function!'


        ### choose the loss function
        if self.vistype == 'binary': ### cross-entropy loss
            L = - T.sum(self.inputs*T.log(self._outputs) + (1-self.inputs)*T.log(1-self._outputs), axis=1)
        elif self.vistype == 'real': # squared-error loss
            L = T.sum(0.5 * ((self.inputs - self._outputs)**2), axis=1)

        # add contractive regularizer
        L = L + self.contraction * contractive_cost

        self._cost = T.mean(L)
        self._grads = T.grad(self._cost, self.params)
        self.cost = theano.function([self.inputs], self._cost)

        self.grad = theano.function([self.inputs], T.grad(self._cost, self.params))
        self.reconstruct = theano.function([self.inputs], self._outputs)
        self.hiddens = theano.function([self.inputs], self._hiddens)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = np.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def get_params(self):
        return np.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        np.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(np.load(filename))

    def normalizefilters(self, center=True):
        def inplacemult(x, v):
            x[:, :] *= v
            return x
        def inplacesubtract(x, v):
            x[:, :] -= v
            return x
        nW = (self.W.get_value().std(0)+SMALL)[np.newaxis, :]
        meannW = nW.mean()
        W = self.W.get_value(borrow=True)
        # CENTER FILTERS
        if center:
            self.W.set_value(inplacesubtract(W, W.mean(0)[np.newaxis,:]), borrow=True)
        # FIX STANDARD DEVIATION
        self.W.set_value(inplacemult(W, meannW/nW),borrow=True)
