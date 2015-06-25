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
import pylab as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import fmin_cg, fmin_bfgs
from optimize import *
from util import *

from autoencoder import * 
from ae_GMD import *

import load_data as ld
from save_util import *
from plot import *
from corrupt_input import *
from basic_utils import *
from tile_img import *


dpath='/u/imdaniel/Documents/machine_learning/'
#dpath='/export/mlrg/imj/machine_learning/'
current_path=dpath+'research/pe_ae/codes/'

'''Trains autoencoder'''
def optimize_ae(train_set, valid_set, test_set, hyper_params, model_type, noise_type, logOn, observeF, wlFlag=False, binaryF=False, symF=False):

    [batch_sz, epsilon, momentum, lam, num_hid, \
                    N, Nv, D, num_epoch, corrupt_in,numpy_rng, activation] = hyper_params

    W,R=init_weight(D,num_hid)
    ae = AutoEncoder(numpy_rng,n_visible=D, n_hidden=num_hid, symF=symF, activation=activation, W=W,R=R)

    hyper_train=[batch_sz, epsilon, momentum, lam,  corrupt_in, model_type, noise_type]
    GMD = GraddescentMinibatch(hyper_train)
    train_model, update_momentum, validate_model, test_model, \
        get_train_mse, get_valid_mse, get_train_divF, get_valid_divF= \
                    GMD.batch_updateX(ae, train_set,valid_set, test_set, wlFlag=wlFlag, binaryF=binaryF)

    n_train_batches = train_set[0].get_value(borrow=True).shape[0]
    n_train_batches /= batch_sz
    n_valid_batches = valid_set[0].get_value(borrow=True).shape[0]
    n_valid_batches /= batch_sz


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 25000  # look as this many examples regardless
    patience_increase = 4 # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    total_time=0;   
    if logOn: train_data = train_set[0].get_value(); valid_data = valid_set[0].get_value(); test_data = test_set[0].get_value()
    logTimes=[]
    lossTests= []; lossValids= []; lossTrains= []
    reconTrains = []; reconValids = [];
    divF_trains =[]; divF_valids = []; WeightLengths=[]
    avg_pos_eigs_train = [];     avg_pos_eigs_valid = []; avg_distI_tr =[]; avg_distI_vl =[]
    avg_distAng_tr =[]; avg_distAng_vl =[]

    if logOn:
        if symF:
            params=[ae.W.get_value(), ae.b.get_value(), ae.b_prime.get_value()]
        else:
            params=[ae.W.get_value(), ae.W_prime.get_value(), ae.b.get_value(), ae.b_prime.get_value()]
        avg_pos_eig_batchi_tr, distAng_tr = sym_analysis(params, train_data[0:25,:], num_hid, symF=False, activation=activation)
        #avg_pos_eig_batchi_vl, distAng_vl = sym_analysis(params, valid_data[0:25,:], num_hid, symF=False, activation=activation)
        avg_pos_eigs_train.append(avg_pos_eig_batchi_tr); avg_distAng_tr.append(distAng_tr);
        #avg_pos_eigs_valid.append(avg_pos_eig_batchi_vl); avg_distAng_vl.append(distAng_vl);

    train_data = train_set[0].get_value()
    [grid_x, grid_y] = get_grid(-0.6,0.6, N=200)                   
    density = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype('float32')
    get_jacobian = get_jacobian_fn(density, ae, density.shape[0])
    start_ae = timeit.default_timer(); 

    optimization_type='cg'
    params = ae.W_prime.get_value(), ae.W.get_value(), ae.b.get_value(), ae.b_prime.get_value()
    params_vec = roll_param(params)
    mag_curls=[]
    while (epoch < num_epoch) and (not done_looping):

        print '...Epoch %d' % epoch
        #error = mean_square_error(params_vec,train_data, D, num_hid)
        #gparams = compute_derivative_ae(params_vec, train_data, D, num_hid)

        for minibatch_index in xrange(n_train_batches): 

            jacobi = get_jacobian()
            curls = get_curl2(jacobi)
            mag_curl = np.sum(curls * curls)
            mag_curls.append(mag_curl)

            #import pdb; pdb.set_trace()
            #visualize_vector(ae,density, train_data,curls=curls, fname=+model_type+'_'+activation+str(epoch)+'_'+str(minibatch_index))

            print '...Optimizing Using ' + optimization_type
            arg = (train_data[minibatch_index*batch_sz:(minibatch_index+1)*batch_sz], D, num_hid,activation)

            if optimization_type=='cg':
                params_vec = list(fmin_cg(mean_square_error, params_vec, \
                        compute_derivative_ae, args=arg, maxiter=10))
            elif optimization_type=='bfgs':
                params_vec = list(fmin_bfgs(mean_square_error, params_vec, \
                        compute_derivative_ae, args=arg, maxiter=10))
            params_vec = roll_param(params)

    plt.plot(np.arange(len(mag_curls)), mag_curls, 'r,')

def run_ae(train_set, valid_set, test_set, hyper_params, model_type, noise_type, logOn, observeF, wlFlag=False, binaryF=False, symF=False):

    [batch_sz, epsilon, momentum, lam, num_hid, \
                    N, Nv, D, num_epoch, corrupt_in,numpy_rng, activation] = hyper_params

    W,R=init_weights(D,num_hid)
    ae = AutoEncoder(numpy_rng,n_visible=D, n_hidden=num_hid, symF=symF, activation=activation, W=W,R=R)

    hyper_train=[batch_sz, epsilon, momentum, lam,  corrupt_in, model_type, noise_type]
    GMD = GraddescentMinibatch(hyper_train)
    train_model, update_momentum, validate_model, test_model, \
        get_train_mse, get_valid_mse, get_train_divF, get_valid_divF= \
                    GMD.batch_updateX(ae, train_set,valid_set, test_set, wlFlag=wlFlag, binaryF=binaryF)

    n_train_batches = train_set[0].get_value(borrow=True).shape[0]
    n_train_batches /= batch_sz
    n_valid_batches = valid_set[0].get_value(borrow=True).shape[0]
    n_valid_batches /= batch_sz


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 25000  # look as this many examples regardless
    patience_increase = 4 # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    total_time=0;   
    if logOn: train_data = train_set[0].get_value(); valid_data = valid_set[0].get_value(); test_data = test_set[0].get_value()
    logTimes=[]
    lossTests= []; lossValids= []; lossTrains= []
    reconTrains = []; reconValids = [];
    divF_trains =[]; divF_valids = []; WeightLengths=[]
    avg_pos_eigs_train = [];     avg_pos_eigs_valid = []; avg_distI_tr =[]; avg_distI_vl =[]
    avg_distAng_tr =[]; avg_distAng_vl =[]

    if logOn:
        if symF:
            params=[ae.W.get_value(), ae.b.get_value(), ae.b_prime.get_value()]
        else:
            params=[ae.W.get_value(), ae.W_prime.get_value(), ae.b.get_value(), ae.b_prime.get_value()]
        avg_pos_eig_batchi_tr, distAng_tr = sym_analysis(params, train_data[0:25,:], num_hid, symF=False, activation=activation)
        #avg_pos_eig_batchi_vl, distAng_vl = sym_analysis(params, valid_data[0:25,:], num_hid, symF=False, activation=activation)
        avg_pos_eigs_train.append(avg_pos_eig_batchi_tr); avg_distAng_tr.append(distAng_tr);
        #avg_pos_eigs_valid.append(avg_pos_eig_batchi_vl); avg_distAng_vl.append(distAng_vl);

    train_data = train_set[0].get_value()
    [grid_x, grid_y] = get_grid(-0.6,0.6, N=200)                   
    density = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype('float32')
    #get_jacobian = get_jacobian_fn(density, ae, density.shape[0])
    get_jacobian = get_jacobian_fn(train_data, ae, train_data.shape[0])
    start_ae = timeit.default_timer(); 
    xaxis = []; mag_curls=[]
    while (epoch < num_epoch) and (not done_looping):

        if epoch % 5 ==0 or epoch < 5:

            jacobi = get_jacobian()
            curls = get_curl2(jacobi)
            log_mag_curl = np.mean(np.sum(curls * curls, axis=1))
            mag_curls.append(log_mag_curl)
            xaxis.append(epoch)
            #import pdb; pdb.set_trace()
            visualize_vector(ae,density, train_data,curls=curls, fname='figs/'+datatype+'/'+model_type+'_'+activation+str(epoch))

        #if epoch == 0: visualize_vector(ae,density, train_data,curls=curls, fname='fcase1_relu/spiral_'+model_type+'_'+activation+str(epoch))
        epoch = epoch + 1

        average_err = []
        for minibatch_index in xrange(n_train_batches): 
            eps = get_epsilon(epsilon, 500, epoch)
            cost_ij = train_model(minibatch_index, lr=eps )
            update_momentum()
            average_err.append(cost_ij)

        if epoch % 5 == 0 or epoch < 8:
            stop_ae = timeit.default_timer()    

            this_validation_loss = validate_model() 
            if observeF:
                reconTrain_batches=[get_train_mse(minibatch_index) for minibatch_index in xrange(n_train_batches)]
                reconTrains.append(np.mean(np.asarray(reconTrain_batches)))
                reconValids.append(get_valid_mse())
                divF_trains_batches=[get_train_divF(minibatch_index) for minibatch_index in xrange(n_train_batches)]
                divF_valids_batches=[get_valid_divF(minibatch_index) for minibatch_index in xrange(n_valid_batches)]
                divF_trains.append(np.mean(np.asarray(divF_trains_batches)))
                divF_valids.append(np.mean(np.asarray(divF_valids_batches)))
                WeightLengths.append(T.mean(ae.W**2).eval())

            # if we got the best validation score until now
            if this_validation_loss <= best_validation_loss and epoch > num_epoch * 0.98:

                # compute zero-one loss on validation set
                if observeF:
                    print 'epoch %i, eps %g, loss train %g, loss valid %g, recon_tr %g, divF_tr %g, divF_vr %g, |W| %g' %\
                        (epoch, eps, np.mean(np.asarray(average_err)), this_validation_loss,\
                                        reconTrains[-1], divF_trains[-1], divF_valids[-1], WeightLengths[-1])
                else: 
                    print 'epoch %i, eps %g, loss train %g, loss valid %g' %\
                        (epoch, eps, np.mean(np.asarray(average_err)), this_validation_loss)
                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                params=[ae.W, ae.W_prime, ae.b, ae.b_prime]
                #save_the_weight(params, current_path+'weights/RhW')
                #save_the_weight(params, current_path+'weights/mnist_'+model_type+'_'+activation+'_'+noise_type+str(corrupt_in)+'')
                best_epoch = epoch  

            elif this_validation_loss > best_validation_loss and epoch > num_epoch * 0.9: #0.50:
                if observeF:
                    print 'epoch %i, eps %g, loss train %g, loss valid %g, recon_tr %g, divF_tr %g, divF_vr %g, |W| %g' %\
                        (epoch, eps, np.mean(np.asarray(average_err)), this_validation_loss,\
                                        reconTrains[-1], divF_trains[-1], divF_valids[-1], WeightLengths[-1])
                else: 
                    print 'epoch %i, eps %g, loss train %g, loss valid %g' %\
                        (epoch, eps, np.mean(np.asarray(average_err)), this_validation_loss)
                done_looping = True 
            else:
                if observeF:
                    print 'epoch %i, eps %g, loss train %g, loss valid %g, recon_tr %g, divF_tr %g, divF_vr %g, |W| %g' %\
                        (epoch, eps, np.mean(np.asarray(average_err)), this_validation_loss,\
                                        reconTrains[-1], divF_trains[-1], divF_valids[-1], WeightLengths[-1])
                else: 
                    print 'epoch %i, eps %g, loss train %g, loss valid %g' %\
                        (epoch, eps, np.mean(np.asarray(average_err)), this_validation_loss)
            if logOn:
                params=[ae.W.get_value(), ae.W_prime.get_value(), ae.b.get_value(), ae.b_prime.get_value()]
                avg_pos_eig_batchi_tr, distAng_tr = sym_analysis(params, train_data[0:25,:], num_hid, symF=False, activation=activation)
                avg_pos_eigs_train.append(avg_pos_eig_batchi_tr); #avg_distI_tr.append(distI_tr);
                avg_distAng_tr.append(distAng_tr);
                #avg_distAng_vl.append(distAng_vl);

            start_ae = timeit.default_timer()

    visualize_vector(ae,density, train_data, curls=curls, fname='figs/'+datatype+'/'+model_type+'_'+activation+str(epoch))
    end_time = time.clock()
    test_loss = test_model()

    print 'Test Loss %g' % test_loss
    plt.figure()
    plt.plot(xaxis, mag_curls, 'r.-')
    plt.savefig('./figs/relu_mag_curl.pdf',bbox_inches='tight') 



    if histF: 
        params=[ae.W.get_value(), ae.W_prime.get_value(), ae.b.get_value(), ae.b_prime.get_value()]
        hist_activation(params, train_data, activation)


    print('optimization complete.')
    print('best validation score of %f %% obtained at iteration %i' %
          (best_validation_loss , best_iter + 1))
    print >> sys.stderr, ('the code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    if logOn: 
        print 'Loss: ', lossTrains, lossValids

        if observeF:
            print 'DivF Train:', divF_trains
            print 'DivF Valid:', divF_valids
            print '|W|:', WeightLengths
            print 'Recon Err TR (MSE):', reconTrains
            print 'Recon Err VL (MSE):', reconValids

    return best_validation_loss , best_epoch, ae, logTimes, test_loss, avg_pos_eigs_train, avg_pos_eigs_valid,\
                        avg_distI_tr, avg_distI_vl, avg_distAng_tr, avg_distAng_vl
               
datatype='line'
histF=False
lam = 0.0
model_type='dae' #**
binaryF=False
dison=False
noise_type='gaussian'
logOn=False
observeF=False
wlFlag=False
activation='sig'
symF=False
num_fold=1
if activation == 'relu': 
    num_hid=1000; 
else: 
    num_hid=1000

if model_type=='ae':
    if activation == 'sig' or activation == 'tanh' :
        if wlFlag: 
            epsilon = 0.06; 
        else: 
            epsilon = 0.0001; 
    elif activation == 'relu':
        if wlFlag:
            epsilon= 0.01
        else:
            epsilon = 0.005;
    momentum=0.0;
elif model_type=='dae':
    if noise_type=='gaussian':
        if activation == 'sig' or activation == 'tanh' :
            if wlFlag: 
                epsilon = 0.06;
            else:
                epsilon = 0.0008#0.0005; 
            corrupt_in=0.05; momentum=0.0; 
        elif activation == 'relu':
            epsilon = 0.01; lam=0.0;batch_sz=50; momentum=0.0; corrupt_in=0.08

    elif noise_type=='salt_pepper':
        if activation == 'sig' or activation == 'tanh':
            if wlFlag: 
                epsilon = 0.06;
            else:
                epsilon = 0.001; 
            corrupt_in=0.1;momentum=0.0
        elif activation == 'relu':
            epsilon = 0.001; lam=0.0;batch_sz=50; momentum=0.0; corrupt_in=0.1
elif model_type=='cae':
    if activation == 'sig' or activation == 'tanh':
        if wlFlag:
            epsilon = 0.06; lam=0.005;
        else:
            epsilon = 0.1; lam=0.0005;
        momentum=0.0;
    elif activation == 'relu':
        if wlFlag:
            epsilon = 0.01; lam=0.0005; 
        else:
            epsilon = 0.01; lam=0.0005; 
        momentum=0.0;batch_sz=50;
    pass

batch_sz=32; num_epoch=500
#if activation=='relu': wlFalg=False
if wlFlag and  model_type != 'cae' : lam=0.0
print 'Display on: %i' % int(dison)
print 'Weight length Constraints : %i' % int(wlFlag)
print 'Activation : %s' % activation


if __name__ == '__main__':

    if datatype=='line':
        data_path = '../synthetic_data/line_data.save'
    elif datatype=='circle':
        data_path = '../synthetic_data/circle_data.save'
    elif datatype=='spiral':
        data_path = '../synthetic_data/spiral_data.save'
    print 'opening data'
    f = open(data_path)
    train_data, valid_data, test_data = pickle.load(f)
    f.close()

    N, D = train_data[0].shape
    Nv = valid_data[0].shape[0]

    numpy_rng=numpy.random.RandomState()
    train_set = ld.shared_dataset(train_data);    
    test_set  = ld.shared_dataset(test_data);    
    valid_set = ld.shared_dataset(valid_data); 

    save = [] 
    sym_test_eucs = [] 
    sym_test_angs = [] 
    logs=[]
    print '---Trained : ' + model_type  + '---Noise Type: ' + noise_type
    for tmp in [1]:
    #for num_epoch in [200]:
    #for corrupt in [0.1,0.2]:
    #for epsilon in [0.01,0.06, 0.03, 0.001]:
    #for epsilon in [0.01,0.05, 0.03]:
    #for epsilon in [0.005, 0.003]:
    #for lam in [0.0005, 0.001, 0.005]:
    #for lam in [0.1, 0.5, 0.05]:
        AvgtestLosses=[]
        for ith_fold in xrange(num_fold):
                if model_type!='dae' : corrupt_in = 0;
                hyper_params = [batch_sz, epsilon, momentum, lam, num_hid, \
                                N, Nv, D, num_epoch, corrupt_in,numpy_rng, activation]

                best_validation_loss , best_epoch, model, logTimes, test_loss, avg_pos_eigs_train, avg_pos_eigs_valid, avg_distAng_tr, avg_distAng_vl, avg_distI_tr, avg_distI_vl = \
                            run_ae(train_set, valid_set, test_set, hyper_params,model_type, noise_type, logOn, observeF, wlFlag, binaryF=binaryF, symF=symF)
                            #optimize_ae(train_set, valid_set, test_set, hyper_params,model_type, noise_type, logOn, observeF, wlFlag, binaryF=binaryF, symF=symF)


                print 'Epsilon %f num_hid %d best epoch %d batch size %d  wc %f, corrupt %f' \
                                % (epsilon, num_hid, best_epoch, batch_sz, lam, corrupt_in)

                print avg_pos_eigs_train, avg_pos_eigs_valid
                print avg_distI_tr, avg_distI_vl
                print avg_distAng_tr, avg_distAng_vl

                NN = len(avg_pos_eigs_train)
                #plt.figure()
                #plt.plot(np.arange(NN), avg_pos_eigs_train, 'r.-')
                #plt.plot(np.arange(NN), avg_pos_eigs_valid, 'r--')
                ##plt.plot(np.arange(NN), np.ones((NN,))*D, 'g-')

                #plt.figure()
                #plt.plot(np.arange(NN), avg_distI_tr, 'r.-')
                #plt.plot(np.arange(NN), avg_distI_vl, 'r--')
                AvgtestLosses.append(test_loss) 
                sym_test_eucs.append(avg_pos_eigs_train)
                sym_test_angs.append(avg_distI_tr)

        print 'Average test loss %g' % (np.mean(np.asarray(AvgtestLosses))) 
        print 'Var test loss %g' % (np.var(np.asarray(AvgtestLosses))) 

        save.append((epsilon, lam, batch_sz, momentum, float(best_validation_loss)))

    print 'euc ',
    print np.mean( np.asarray(sym_test_eucs, dtype='float32'),axis=0)
    print np.std( np.asarray(sym_test_eucs, dtype='float32'),axis=0)
    print 'anlge '
    print np.mean( np.asarray(sym_test_angs, dtype='float32'),axis=0)
    print np.std( np.asarray(sym_test_angs, dtype='float32'),axis=0)
    print 'euc identity'
    #print np.mean( np.asarray(sym_test_angs, dtype='float32'),axis=0)
    #print np.std( np.asarray(sym_test_angs, dtype='float32'),axis=0)


    print save
    print logs

    print '---Trained : ' + model_type  + '---Noise Type: ' + noise_type


