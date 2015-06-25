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
import timeit, time, pickle, sys, os, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from contractiveUAE import cUAE
from contractiveAE import cAE
from train import GraddescentMinibatch
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
if __name__ == '__main__':

    from vector_field_sim import (VectorField,
                                  visualize_vector_field,
                                  visualize_contours)
    import theano
    import numpy as np
    import cPickle as ld

    ########################################
    ## Stage 0: Defining Hyper Parameters ##
    ########################################

    ## Shuffle data ?
    shuffle = True
    ## Visualise results?
    visualise = True

    ## Number of hidden units
    n_hiddens = 1000
    contraction = 2.00
    activation = 'relu'
    BINARY = False
    corruptF = True

    epochs = 500
    learningrate = 0.0005
    MINIBATCH_SIZE = 1000
    momentum = 0.00
    normalizefilters = False
    rng = None #np.random.RandomState(123)
    verbose = True
    eps = 1e-8
    ## Number of point in each dimension to sample
    n_points = 100  ## square for the number of examples. (i.e: 100 in x, 100 in y,
    ## 10000 overall)

    ###############################
    ## Stage 1: Experiment setup ##
    ###############################

    ##,---------------------------------
    ##| Defining vector field parameters
    ##`---------------------------------


    syntheticField = VectorField()

    ##,---------------------------------
    ##| Computing synthetic Vector Field
    ##`---------------------------------

    syntheticField.compute_field(n_points=n_points)

    ##,--------------------------
    ##| Sampling the vector field
    ##`--------------------------

    sampled_dic = syntheticField.sample_field()
    sampled_input = sampled_dic['axes']
    sampled_vector_field = sampled_dic['density']
    sampled_grad_field = sampled_dic['vector_field']
    sampled_grad_curl_free = sampled_dic['curl_free']
    sampled_grad_div_free = sampled_dic['div_free']

    ##,--------------
    ##| Preprocessing
    ##`--------------
    ##scaling = MinMaxScaler((0.1, 0.9))
    data_input = sampled_grad_field #sampled_grad_curl_free

    ##,---------------------------
    ##| Preparing training dataset
    ##`---------------------------

    if shuffle:
        rows_idx = np.arange(sampled_input.shape[0])
        np.random.shuffle(rows_idx)
        train_data = [sampled_input[rows_idx, :],
                      data_input[rows_idx, :]]
    else:
        train_data = [sampled_input, data_input]


    #######################
    ## Stage 2: Training ##
    #######################

    train_x, train_F = train_data
    data = train_F
    ## Sharing training data
    data_s = theano.shared(np.asarray(data,
                                      dtype=theano.config.floatX),
                           borrow=True)
    n_visible = train_F.shape[1]
    n_samples = train_F.shape[0]

    ##,--------------------
    ##| Instantiating model
    ##`--------------------

    model = cAE(n_visible=n_visible, n_hiddens=n_hiddens,
                contraction=contraction,
                activation=activation, n_samples=n_samples,
                BINARY=BINARY, MINIBATCH_SIZE=MINIBATCH_SIZE, corruptF=corruptF)

    ##,----------------------
    ##| Instantiating trainer
    ##`----------------------

    gd = GraddescentMinibatch(model=model, data=data_s,
                              batchsize=MINIBATCH_SIZE,
                              learningrate=learningrate,
                              momentum=momentum,
                              normalizefilters=normalizefilters,
                              rng=rng,
                              verbose=verbose)

    ##,---------
    ##| Training
    ##`---------

    #for epoch in np.arange(epochs):
    #    gd.step()
    while (model.cost(data_s.get_value()) > eps) and gd.epochcount < epochs:
        gd.step()

    ##################################
    ## Stage 4: Visualizing results ##
    ##################################

    ##,-----------------------
    ##| Getting reconstruction
    ##`-----------------------
    scaled_input = np.float32(sampled_input)
    #scaled_input = np.float32(sampled_input)
    F_rec = model.reconstruct(scaled_input) - scaled_input

    ##,---------------------------
    ##| Visualizing reconstruction
    ##`---------------------------
    F_truth = sampled_grad_field - sampled_input
    grid_x_hat, grid_y_hat = scaled_input.T.reshape((2, n_points, n_points))
    grid_x, grid_y = np.float32(sampled_input).T.reshape((2, n_points, n_points))
    dx, dy = F_truth.T.reshape((2, n_points, n_points))
    dx_hat, dy_hat = F_rec.T.reshape((2, n_points, n_points))
    p = np.float32(sampled_vector_field).T.reshape((n_points, n_points))

    ## Visualizing reconstruction
    viz_fun = visualize_vector_field
    ##viz_fun = visualize_contours

    if visualise:
        viz_fun(grid_x_hat, grid_y_hat,
                dx_hat, dy_hat,
                p=p, title="Reconstructed Vector Field")
        viz_fun(grid_x, grid_y,
                dx, dy,
                p=p, title="ground truth")
        # for mode in ['field']:
        #     for part in ['complete', 'curl_free', 'div_free']:
        #         syntheticField.visualize_field(part=part, mode=mode)

