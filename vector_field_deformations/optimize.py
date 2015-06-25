import numpy as np
import scipy as sp
from util import *

'''Computes the derivative of auto-encoders with respective to the parameters'''
def compute_derivative_ae(params_vec, data,D,num_hid, act_type='sig'):

    R, W, hbias, vbias = unroll_params(params_vec, D,num_hid)
    preH =np.dot(data, W) + hbias
    if act_type=='tanh':
        h = np.tanh(preH)
    else:
        h = sigmoid(preH)
    y = np.dot(h,R)
    error = (y - data)**2

    if act_type=='tanh':
         h_prime = (1- h*h)  
    else:
        h_prime = h * (1-h)

    dEdR = 2*np.dot(error.T, h_prime).T
    dEdh = 2*np.dot(R, error.T) * h_prime.T
    dEdW = np.dot(dEdh, data).T

    dEdvbias = 2*np.sum(error.T,axis=1)
    dEdhbias = 2*np.sum(dEdh ,axis=1)
 
    return roll_param([dEdR, dEdW, dEdhbias, dEdvbias])

'''Propagate forward in Neural network (auto-encoder) '''
def forward_prop(params, data, act_type='sig'):

    R, W, hbias, vbias = params
    preH =np.dot(data, W) + hbias
    if act_type=='tanh':
        h = np.tanh(preH)
    else:
        h = sigmoid(preH)
    y = np.dot(h,R)

    return y, h

'''Computes the means square error'''
def mean_square_error(params_vec,data,D, num_hid, act_type='sig' ):

    R, W, hbias, vbias = unroll_params(params_vec, D, num_hid)
    params = R, W, hbias, vbias
    y, h = forward_prop(params, data, act_type=act_type)
    error = (y - data)**2

    #print 'Mean Square Error %g' % np.mean(error)
    return np.mean(np.sum(error, axis=1))

'''unroll the parameter to a vector (function name is reversed with rolled one)'''
def roll_param(params):
   return np.hstack((params[0].flatten(), params[1].flatten(), \
            params[2], params[3]))

'''roll the parameter to a vector  '''
def unroll_params(params_vec, D,H):
    R = params_vec[0:D*H].reshape((H,D))
    W = params_vec[D*H:(D*H)+D*H].reshape((D,H))
    hbias = params_vec[(2*D*H):(2*D*H)+H]
    vbias = params_vec[(2*D*H)+H:]
    return [R,W,hbias, vbias] 


