#Copyright (c) 2013, Roland Memisevic
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
''' A collection of functions for training and inference '''
import numpy
import theano
import theano.tensor as T

MOMENTUM = 0.9

def loss_batchwise(data, model, batchsize):
        numbatches = data.n_samples / batchsize
        cost = 0
        for batch in range(numbatches):
            if numbatches>100:
                if batch % 100 == 0:
                    print batch
            givens={model.inputs:data._inputs[batch*batchsize:(batch+1)*batchsize],model.labels:data._labels[batch*batchsize:(batch+1)*batchsize]}
            cost_f = theano.function([],model.get_cost(), givens=givens)
            cost+=cost_f()
        return cost/numbatches

def error_rate_batchwise(data, model, batchsize):
        numbatches = data.n_samples / batchsize
        error = 0
        for batch in range(numbatches):
            if numbatches>100:
                if batch % 50 == 0:
                    print batch
            givens={model.inputs:data._inputs[batch*batchsize:(batch+1)*batchsize],model.labels:data._labels[batch*batchsize:(batch+1)*batchsize]}
            error_f = theano.function([],model.get_error_rate(), givens=givens)
            error+=error_f()

        return error/numbatches

def error_rate(inputs, labels, model):
    givens={model.inputs:inputs,model.labels:labels}
    error_f = theano.function([],model.get_error_rate(), givens=givens)
    return error_f()

class GraddescentMinibatchLabeled(object):

    def __init__(self, model, data, labels, batchsize, learningrate, momentum = MOMENTUM, normalizefilters=False, numcases=None, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.labels        = labels
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self._cost = 0
        if numcases is None:
                ##self.numbatches = self.data.get_value().shape[0] / batchsize
                self.numbatches = self.data.shape[0] / batchsize
        else:
            self.numbatches = numcases / batchsize
        self.momentum      = momentum
        self.normalizefilters = normalizefilters
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar()
        self.incs = \
          dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, dtype=numpy.float32), name='inc_'+p.name)) for p in self.model.params])

        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n

        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad
            self.updates[_param] = _param + self.incs[_param]
        ##OVERRIDE HID LEARNINGRATE TO BE SMALLER
        #self.inc_updates[self.incs[self.model.layer.whf]] = self.momentum * self.incs[self.model.layer.whf] - 0.01*self.learningrate * self.model._grads[2]
        #self.updates[self.model.layer.whf] = self.model.layer.whf + self.incs[self.model.layer.whf]

        self._updateincs = theano.function([self.index], self.model._cost, updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize], #cut out minibatch
                          self.model.labels:self.labels[self.index*self.batchsize:(self.index+1)*self.batchsize]})

        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self, crossvalidate = False):
        cost = 0.0
        stepcount = 0.0
        index = self.rng.permutation(self.numbatches-1)
        #print index
        if crossvalidate:
            crossvalid_index = index[-1]
            index = index[:-1]
            #print crossvalid_index

        for batch_index in index:
        #for batch_index in range(self.numbatches):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)

        self.epochcount += 1


        if crossvalidate:
            stepcount = 1.0
            #print 'crossvalidate'
            cdrossvalid_cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(crossvalid_index)
            crossvalid_error =  error_rate(inputs = self.data[crossvalid_index*self.batchsize:(crossvalid_index+1)*self.batchsize],
                                           labels=self.labels[crossvalid_index*self.batchsize:(crossvalid_index+1)*self.batchsize], model=self.model)
        if self.verbose:
            print 'current: epoch %d, cost: %f' % (self.epochcount, cost)

        if self.normalizefilters:
            self.model.normalizefilters()

        if crossvalidate:
            return cost, crossvalid_error, cdrossvalid_cost
        else:
            return cost

class GraddescentMinibatch(object):

    def __init__(self, model, data, batchsize, learningrate, momentum = MOMENTUM, normalizefilters=False, numcases=None, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self._cost = 0
        if numcases is None:
            self.numbatches = self.data.get_value().shape[0] / batchsize
        else:
            self.numbatches = numcases / batchsize
        self.momentum      = momentum
        self.normalizefilters = normalizefilters
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar()
        self.incs = \
          dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, dtype=numpy.float32), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n

        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad
            self.updates[_param] = _param + self.incs[_param]
        ##OVERRIDE HID LEARNINGRATE TO BE SMALLER
        #self.inc_updates[self.incs[self.model.layer.whf]] = self.momentum * self.incs[self.model.layer.whf] - 0.01*self.learningrate * self.model._grads[2]
        #self.updates[self.model.layer.whf] = self.model.layer.whf + self.incs[self.model.layer.whf]

        self._updateincs = theano.function([self.index], self.model._cost, updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize] })#cut out minibatch


        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self, crossvalidate = False):
        cost = 0.0
        stepcount = 0.0
        index = self.rng.permutation(self.numbatches-1)
        #print index


        for batch_index in index:
        #for batch_index in range(self.numbatches):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)

        self.epochcount += 1


        if self.verbose:
            print 'current: epoch %d, cost: %f' % (self.epochcount, cost)

        if self.normalizefilters:
            self.model.normalizefilters()

        return cost



