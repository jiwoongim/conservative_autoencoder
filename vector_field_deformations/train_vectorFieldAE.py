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
import timeit, pickle, sys
import theano
import theano.tensor as T
import os
import signal, sys
import timeit, time

from util import *

default_path = '../../../implementation/'
path = os.path.abspath(os.path.join(os.path.dirname(__file__), default_path+'util/'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
import load_data as ld
from save_util import *
from plot import *
from corrupt_input import *
from basic_utils import *
from tile_img import *

from genVectorFieldDataset import *
from vectorFieldAE import *
from GMD_vectorFieldAE import *

#dpath='/u/imdaniel/Documents/machine_learning/'
dpath='/export/mlrg/imj/machine_learning/'
current_path=dpath+'research/pe_ae/codes/'

def run_ae(train_set, hyper_params, wlFlag=False, binaryF=True):

    [batch_sz, epsilon, momentum, lam, num_hid, \
                                N, D, num_epoch, numpy_rng, activation] = hyper_params
    ae = VectorFieldAE(numpy_rng,n_visible=D, n_hidden=num_hid,symF=True, activation=activation)

    hyper_train=[batch_sz, epsilon, momentum, lam]
    GMD = GraddescentMinibatch(hyper_train)
    train_model, update_momentum = \
                    GMD.batch_updateX(ae, train_set,wlFlag=wlFlag, binaryF=binaryF)

    n_train_batches = train_set[0].get_value(borrow=True).shape[0]
    n_train_batches /= batch_sz


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    start_time = time.clock()
    
    epoch = 0
    start_ae = timeit.default_timer(); 
    logTimes=[]; lossTrains= []; 

    for epoch in xrange(num_epoch):
        epoch = epoch + 1
        average_err = []
        for minibatch_index in xrange(n_train_batches): 
            eps = get_epsilon(epsilon, 500, epoch)
            cost_ij = train_model(minibatch_index, lr=eps )
            update_momentum()
            average_err.append(cost_ij)


        if epoch % 5 == 0 or epoch == 1:
            stop_ae = timeit.default_timer()    
            lossTrains.append(np.mean(np.asarray(average_err)))
            print 'epoch %i, eps %g, loss train %g' % (epoch, eps, lossTrains[-1])
            start_ae = timeit.default_timer()
            
    end_time = time.clock()

    print('optimization complete.')
    print >> sys.stderr, ('the code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return ae, lossTrains


def free_energy(params, X):
    ''' Function to compute the free energy '''

    W,hbias,vbias = params
    wx_b = np.dot(X, W) + hbias
    vbias_term = np.dot(X, vbias)
    hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
    #entropy = - np.sum(X * np.log(X) + (1-X) * np.log(1 - X), axis=1)
    return hidden_term + vbias_term #+ entropy


def evaluate(model, X):
    
    params = [model.W.get_value(), model.b.get_value(), model.b_prime.get_value()]
    E0 = free_energy(params, X[0])
    E1 = free_energy(params, X[1])
    return np.mean((E0>E1))
    #return np.sum((E0 > E1)) / float(X[0].shape[0])


binaryF=False
dison=False
wlFlag=True
activation='sig'
num_fold=1
num_hid=500

if activation == 'relu':
    epsilon = 0.003; lam=0.0; momentum=0.5;batch_sz=50; num_epoch=300#1000 0.0001 
else:
    epsilon = 0.006; lam=0.0; momentum=0.5;batch_sz=50; num_epoch=200#1000 0.0001 

if wlFlag: lam=0.0


if __name__ == '__main__':

    #train_data = ld.unpickle(current_path+'UniformVectField.save')
    #train_data = ld.unpickle(current_path+'DataVectField.save')
    train_data = ld.unpickle(current_path+'NoiseDataVectField.save')
    N, D = train_data[0].shape

    numpy_rng=numpy.random.RandomState(123)
    train_set = [ theano.shared(np.asarray(train_data[0], dtype=theano.config.floatX)),\
                  theano.shared(np.asarray(train_data[1], dtype=theano.config.floatX))]


    save = [] 
    logs=[]


    data_path = dpath+'data/MNIST/mnist_binary.pkl'
    print 'opening data'
    f = open(data_path)
    Eval_data,tmp,tmp  = pickle.load(f)
    Eval_data0 = Eval_data[0]
    f.close()

    #for epsilon in [0.003, 0.001,0.0001]:
    #for epsilon in [0.006, 0.003,0.001]:
    for tmp in [1]:
        AvgtestLosses=[]
        for ith_fold in xrange(num_fold):

            hyper_params = [batch_sz, epsilon, momentum, lam, num_hid, \
                            N, D, num_epoch, numpy_rng, activation]

            model= run_ae(train_set, hyper_params, wlFlag, binaryF )

            print 'Epsilon %f num_hid %d batch size %d  wc %f' \
                            % (epsilon, num_hid, batch_sz, lam)

            #Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.9)
            #score9 = evaluate(model, [Eval_data0, Eval_data1])
            #Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.8)
            #score8 = evaluate(model, [Eval_data0, Eval_data1])
            #Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.7)
            #score7 = evaluate(model, [Eval_data0, Eval_data1])
            #Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.6)
            #score6 = evaluate(model, [Eval_data0, Eval_data1])
            Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.5)
            score5 = evaluate(model, [Eval_data0, Eval_data1])
            Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.4)
            score4 = evaluate(model, [Eval_data0, Eval_data1])
            Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.3)
            score3 = evaluate(model, [Eval_data0, Eval_data1])
            Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.2)
            score2 = evaluate(model, [Eval_data0, Eval_data1])
            Eval_data1 = getNoisyPoints(numpy_rng, Eval_data0, corrupt_in=0.1)
            score1 = evaluate(model, [Eval_data0, Eval_data1])
            print 'Scores:' 
            print [score5,score4,score3,score2,score1]
            #print [score9,score8,score7, score6,score5,score4,score3,score2,score1]

        save.append((epsilon, lam, batch_sz, momentum))

    print save
    print logs

    #params=[model.W.get_value(), model.b.get_value(), model.b_prime.get_value()]
    #get_num_pos_eigen(params, valid_data[0][0:50,:], num_hid,symF=True)

    if dison:
        display_weight(model.W.get_value().T, 'AE W', 100, fname='./figs/AE_W')

        X = train_set[0][0:16,:].eval()
        reconX = model.get_reconstructed_input_given_x(X, binary=binaryF ).eval()
        display_dataset(X     , (28,28), (4,4), i=2)
        display_dataset(reconX, (28,28), (4,4), i=2)


        plt.figure()
        display_weight(model.W.get_value(borrow=True).T, 'no corruption', 3)

    plt.show() 






