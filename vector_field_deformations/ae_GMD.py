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
import cPickle

import theano
import theano.tensor as T
import theano.tensor.signal.conv 
from theano.tensor.shared_randomstreams import RandomStreams


class GraddescentMinibatch(object):
    """ Gradient descent trainer class. """

    def __init__(self, hyper_params):

        self.batch_sz, self.epsilon, self.momentum, self.lam, self.corrupt,\
                            self.model_type,self.noise_type = hyper_params


    def batch_updateX(self, model, train_set, valid_set, test_set, wlFlag=False, binaryF=True):

        update_grads = []; updates_mom = []; deltaWs = {}
        X = T.fmatrix('X');  
        mom = T.scalar('mom'); index = T.iscalar('i'); lr = T.fscalar('lr');

        print self.noise_type, self.corrupt
        if self.model_type=='cae':
            cost = model.cost(X, self.corrupt, binary=binaryF) \
                + self.lam * model.contractive_pen(X, self.batch_sz) 
        else:
            cost = model.cost(X, self.corrupt, binary=binaryF, ntype=self.noise_type)\
                    + self.lam * model.weight_decay()

        cost_test = model.cost(X, binary=binaryF) 
        gparams = T.grad(cost, model.params)

        #Update momentum
        for param in model.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            deltaWs[param] = theano.shared(init)

        for param in model.params:
            updates_mom.append((param, param + deltaWs[param] * \
                            T.cast(mom, dtype=theano.config.floatX)))       

        for param, gparam in zip(model.params, gparams):

            deltaV = T.cast(mom, dtype=theano.config.floatX)\
                    * deltaWs[param] - gparam * T.cast(lr, dtype=theano.config.floatX)     #new momentum

            update_grads.append((deltaWs[param], deltaV))

            new_param = param + deltaV
            fixed_length = T.sqrt(10)
            if wlFlag and param.get_value().ndim > 1 and param.name == 'W':
                len_W = T.sqrt(T.sum(new_param**2, axis=0))
                desired_W = T.clip(len_W, 0., fixed_length)
                ratio = desired_W  / (len_W + 1e-7)
                new_param = new_param * ratio
            if wlFlag and param.get_value().ndim > 1 and param.name == 'R':
                len_R = T.sqrt(T.sum(new_param.T**2, axis=0))
                desired_R = T.clip(len_R, 0., fixed_length)
                ratio = desired_R  / (len_R + 1e-7)
                new_param = (new_param.T * ratio).T
            update_grads.append((param, new_param))       
        
        if wlFlag: print 'Fixed lenght : %g' % fixed_length.eval()
        update_momentum = theano.function([theano.Param(mom,default=self.momentum)],\
                                                        [], updates=updates_mom)

        train_update    = theano.function([index,theano.Param(lr,default=self.epsilon),\
                theano.Param(mom,default=self.momentum)], outputs=cost, updates=update_grads,\
                    givens={ X:train_set[0][index * self.batch_sz: \
                                            (index + 1) * self.batch_sz]})

        get_valid_cost   = theano.function([], outputs=cost_test,\
                            givens={ X:valid_set[0]})

        get_test_cost    = theano.function([], outputs=cost_test,\
                            givens={ X:test_set[0]})

        mse = model.get_reconstructed_err(X)
        get_train_mse    = theano.function([index], outputs=mse,\
                            givens={ X:train_set[0][index * self.batch_sz: \
                                            (index + 1) * self.batch_sz]})

        get_valid_mse    = theano.function([], outputs=mse,\
                            givens={ X:valid_set[0]})

        divF = model.divFreeEnergy(X, self.batch_sz)
        get_train_divF    = theano.function([index], outputs=divF,\
                            givens={ X:train_set[0][index * self.batch_sz: \
                                            (index + 1) * self.batch_sz]})

        get_valid_divF    = theano.function([index], outputs=divF,\
                            givens={ X:valid_set[0][index * self.batch_sz: \
                                            (index + 1) * self.batch_sz]})


        return train_update, update_momentum, get_valid_cost, get_test_cost,\
                get_train_mse, get_valid_mse, get_train_divF, get_valid_divF


    def batch_updateX2(self, model, train_set, valid_set, test_set, wlFlag=False, binaryF=True):

        update_grads = []; updates_mom = []; deltaWs = {}
        X = T.fmatrix('X');  
        mom = T.scalar('mom'); index = T.iscalar('i'); lr = T.fscalar('lr');

        print self.noise_type, self.corrupt
        if self.model_type=='cae':
            cost = model.cost(X, self.corrupt, binary=binaryF) \
                + self.lam * model.contractive_pen(X, self.batch_sz) 
        else:
            cost = model.cost(X, self.corrupt, binary=binaryF, ntype=self.noise_type)\
                    + self.lam * model.weight_decay()

        cost_test = model.cost(X, binary=binaryF) 
        gparams = T.grad(cost, model.params)

        #Update momentum
        for param in model.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            deltaWs[param] = theano.shared(init)

        for param in model.params:
            updates_mom.append((param, param + deltaWs[param] * \
                            T.cast(mom, dtype=theano.config.floatX)))       

        for param, gparam in zip(model.params, gparams):

            deltaV = T.cast(mom, dtype=theano.config.floatX)\
                    * deltaWs[param] - gparam * T.cast(lr, dtype=theano.config.floatX)     #new momentum

            update_grads.append((deltaWs[param], deltaV))

            new_param = param + deltaV
            fixed_length = T.sqrt(10)
            if wlFlag and param.get_value().ndim > 1 and param.name == 'W':
                len_W = T.sqrt(T.sum(new_param**2, axis=0))
                desired_W = T.clip(len_W, 0., fixed_length)
                ratio = desired_W  / (len_W + 1e-7)
                new_param = new_param * ratio
            if wlFlag and param.get_value().ndim > 1 and param.name == 'R':
                len_R = T.sqrt(T.sum(new_param.T**2, axis=0))
                desired_R = T.clip(len_R, 0., fixed_length)
                ratio = desired_R  / (len_R + 1e-7)
                new_param = (new_param.T * ratio).T
            update_grads.append((param, new_param))       
        
        if wlFlag: print 'Fixed lenght : %g' % fixed_length.eval()
        update_momentum = theano.function([theano.Param(mom,default=self.momentum)],\
                                                        [], updates=updates_mom)

        train_update    = theano.function([index,theano.Param(lr,default=self.epsilon),\
                theano.Param(mom,default=self.momentum)], outputs=cost, updates=update_grads,\
                    givens={ X:train_set[0][index * self.batch_sz: \
                                            (index + 1) * self.batch_sz]})

        get_valid_cost   = theano.function([], outputs=cost_test,\
                            givens={ X:valid_set[0]})

        get_test_cost    = theano.function([], outputs=cost_test,\
                            givens={ X:test_set[0]})

        mse = model.get_reconstructed_err(X)
        get_train_mse    = theano.function([index], outputs=mse,\
                            givens={ X:train_set[0][index * self.batch_sz: \
                                            (index + 1) * self.batch_sz]})

        get_valid_mse    = theano.function([], outputs=mse,\
                            givens={ X:valid_set[0]})

        return train_update, update_momentum, get_valid_cost, get_test_cost



