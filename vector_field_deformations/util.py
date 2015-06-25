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
import scipy as sc
import timeit, pickle, sys
import theano
import theano.tensor as T
import os
import signal, sys
import timeit, time
import matplotlib 
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt


TINY=1.19209e-07

'''Log of Sigmoid'''
def log_sigmoid(x):
    m = np.maximum(-x,0)
    return -(np.log(np.exp(-m) + np.exp(-x-m)) + m)

'''Sigmoid'''
def sigmoid(x):
    return np.exp(log_sigmoid(x))

'''Distance from X to X.T'''
def compute_sym(X):    
    return np.sum((X-X.T)*(X-X.T)) #/ (X*X).sum().sum()

'''Distance from X to Identity matrix'''
def compute_distI(X):
    I = np.eye((X.shape[0]))
    return np.sqrt(np.sum((X-I)*(X-I)))

'''DIstance of symmetricity'''
def compute_symdist(X):
    Y = (X + X.T)/2
    return np.sqrt( np.sum(Y*Y) / np.sum(X*X) )

'''Computes the length of bias'''
def get_bias_length(b):
    return np.sqrt(np.sum(b*b))

'''Counts number of eigenvalues that have only real numbers'''
def get_num_eigen(drdxi):
    eigvalue = np.linalg.eigvals(drdxi)

    imgE = np.imag(eigvalue)
    ind_real = np.argwhere(imgE == 0)

    indices = (imgE < TINY) * (imgE > -1*TINY)
    return np.sum( (imgE == 0))

'''Feedforward propgation in one hidden layer neural network (auto-encoder)'''
def feedforward(activation, x, params, zerobiasF=False):

    W,R,hbias,vbias = params

    if zerobiasF:
        z = np.dot(x, W)
        h = (z > 1.0) * z 
    elif activation == 'relu': 
        z = np.dot(x, W) + hbias
        h = np.maximum(0,z)
        indices = np.argwhere(h>0)
        dh = np.zeros((h.shape[0]))
        dh[indices] = 1
        dh = np.diag(dh)
    elif activation == 'tanh':
        z = np.dot(x, W) + hbias
        h  = np.tanh(z)
        dh = np.diag(1 - np.tanh(z) * np.tanh(z))
    else:
        z = np.dot(x, W) + hbias
        h  = sigmoid(z)
        dh = np.diag(h*(1-h)) 

    return h, dh

'''Returns histogram of hidden activations of one hidden layer neural network'''
def hist_activation(params, X, activation):

    W,R,hbias,vbias = params
    hid_act,dh = feedforward(activation, X, params)

    hid = hid_act.flatten()

    if activation =='tanh':
         bins= (np.arange(100) *2 - 100.0) / 100.0       
    else:
        bins= np.arange(100) / 100.0
    
    plt.figure()
    n, bins, patches = pl.hist(hid, bins, normed=1, histtype='bar', rwidth=0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #plt.savefig('histo_sig_wl.png', bbox_inches='tight') 


'''Weight initialization for neural net.'''
def init_weights(D,H):

    W = np.random.uniform( low = -1, high = 1, size=(D, H)).astype('float32')
    R = np.random.uniform( low = -1, high = 1, size=(H, D)).astype('float32')
    return theano.shared(value=W, name='W'), theano.shared(value=R, name='W')


def get_grid(A,B, N=200):
    '''Uniform grid N by N from A to B'''
    
    axis1 = (B-A) * np.arange(N) / float(N) +A
    axis2 = (B-A) * np.arange(N) / float(N) +A
    grid_x = np.zeros((N,N));
    grid_y = np.zeros((N,N))
    for i in xrange(N):
        for j in xrange(N):
            grid_x[i,j]=float(axis1[i])
            grid_y[i,j]=float(axis1[j])

    return [grid_x, grid_y]

'''Visualize a vector field in two dimensions.'''
def visualize_vector_field(x, y, dx, dy, p,datapoints,curls=None, fname='tmp', **kwargs):

    title = 'Vector Field: '
    if 'title' in kwargs:
        title += kwargs['title']

    skip = (slice(None, None, 5), slice(None, None, 5))

    plt.figure()
    fig, ax = plt.subplots()    
    plt.quiver(x[skip], y[skip], dx[skip], dy[skip], p[skip],  color='r')
    plt.plot(datapoints[:,0], datapoints[:,1], 'g.')
    #ax.set(aspect=1, title=title)
    ax.set_xlim([-0.6,0.6])
    ax.set_ylim([-0.6,0.6])
    plt.colorbar()
    plt.savefig('./'+fname,bbox_inches='tight') 

    if False and curls is not None: ## Commented out for now
        N = int(np.sqrt(curls.shape[0]))   
        cgrid_x = curls[:,0].T.reshape((N, N))
        cgrid_y = curls[:,1].T.reshape((N, N))
        cdgrid_x = curls[:,0].T.reshape((N, N)) - cgrid_x
        cdgrid_y = curls[:,1].T.reshape((N, N)) - cgrid_y


        plt.figure()
        fig, ax = plt.subplots()       
        plt.quiver(cgrid_x[skip], cgrid_y[skip], cdgrid_x[skip], cdgrid_y[skip], p[skip],  color='b')
        plt.plot(datapoints[:,0], datapoints[:,1], 'g.')
        ax.set(aspect=1, title="Curl "+title)
        plt.savefig('./vector_field_visualization/'+'curl_'+fname,bbox_inches='tight') 

def visualize_vector(model, density, train_data, curls=None, fname='tmp'):
    
    N = int(np.sqrt(density.shape[0]))
    density_mapping = model.get_reconstructed_input_given_x(density, binary=False).eval()
    p = np.mean((density_mapping - density)**2, axis=1).reshape((N,N))

    density_mapping = density_mapping - density
    grid_x = density[:,0].T.reshape((N, N))
    grid_y = density[:,1].T.reshape((N, N))
    grid_rx = density_mapping[:,0].T.reshape((N, N))
    grid_ry = density_mapping[:,1].T.reshape((N, N))
    #p=np.ones((grid_x.shape[0],grid_y.shape[0]))


    viz_fun = visualize_vector_field
    viz_fun(grid_x, grid_y,
            grid_rx, grid_ry,
            p=p, curls=curls, datapoints=train_data, fname=fname,title="r(x)-x Vector Field")

'''Computes curl of auto-encoder's Jacobian Matrix '''
def get_curl2(jacobian):
    N,D,D = jacobian.shape
    curls = np.zeros((N,D))
    curls[:,0] = jacobian[:,0,1]
    curls[:,1] = -jacobian[:,1,0]
    return curls

'''Computes the Jacobian'''
def get_jacobian_fn(data, model, N, binaryF=False):

    X = T.fmatrix('X'); 
    hhh = model.get_hidden_values(X)
    DHW = model.get_jacobian(hhh,model.W,N) 
    jacobian = T.dot(DHW,model.W_prime)
    #diag = -T.eye(N) + T.ones((N)) 
    #(diag * jacobian).
    return theano.function([], jacobian, givens={X:data})

'''Symmetricity Analysis'''
def sym_analysis(params, data, dimH, symF=False, zerobiasF=False, activation='sig'):

    N = data.shape[0]
    if symF:
        if zerobiasF:
            W = params[0]
        else:
            W,hbias,vbias = params
        R = W.T
    else:
        if zerobiasF:
            W,R = params
        else:
            W,R,hbias,vbias = params

    tot_distIs = []
    tot_eig = []
    tot_distSym = []
    for i in xrange(N):
        h,dh= feedforward(activation, data[i,:], [W,R,hbias,vbias], zerobiasF=zerobiasF)          

        A  = np.dot(dh,R)
        drdxi = np.dot(W,A)

        bias_length = get_bias_length(hbias)
        tot_eig.append(bias_length)
        
        distSym = compute_symdist(drdxi)
        tot_distSym.append(distSym)

    avg_eig = np.mean(np.asarray(tot_eig))
    avg_distI = np.mean(np.asarray(tot_distIs))
    avg_distSym = np.mean(np.asarray(tot_distSym))


    return avg_eig, avg_distSym#, avg_distI

def sym_analysis2(params, data, dimH, symF=False, zerobiasF=False, activation='sig'):

    N = data.shape[0]
    W0,W1,R,hbias0,hbias1,vbias = params

    tot_distIs = []
    tot_eig = [0]
    tot_distAng = []
    for i in xrange(N):
        z1 = sigmoid(np.dot(data[i,:], W0) + hbias0)
        z2 = sigmoid(np.dot(z1, W1) + hbias1)
        dh1 = np.diag(z1*(1-z1)) 
        dh2 = np.diag(z2*(1-z2)) 

        A  = np.dot(dh1,np.dot(W1, np.dot(dh2,R)))
        drdxi = np.dot(W0,A)
        
        distAng = compute_symdist(drdxi)
        tot_distAng.append(distAng)

    avg_eig = np.mean(np.asarray(tot_eig))
    #avg_distI = np.mean(np.asarray(tot_distIs))
    avg_distAng = np.mean(np.asarray(tot_distAng))


    return avg_eig, avg_distAng#, avg_distI






