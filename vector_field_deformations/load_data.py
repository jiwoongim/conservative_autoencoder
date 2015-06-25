''' Version 1.000

 Code provided by Jiwoong Im

 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

import cPickle, gzip, numpy
import theano
import theano.tensor as T
import numpy as np 
import math


def save_the_weight(x,fname):
    f = file(fname+'.save', 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def separate_data_into_classes(train_set, num_classes, flag=1):
   
    sep_train_set = []
    num_cases_per_class = []

    for class_i in xrange(num_classes):
        train_data = train_set[0][train_set[1]==class_i,:]
        Nc = train_data.shape[0]
        num_cases_per_class.append(Nc) 

        if flag:
            sep_train_set.append(shared_dataset([train_data, class_i *np.ones((Nc,1),dtype='float32')]))
        else:
            sep_train_set.append([train_data, class_i *np.ones((Nc,1),dtype='float32')])
    return sep_train_set, num_cases_per_class



def load_dataset(path):
    # Load the dataset
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return [train_set[0], train_set[1]], \
            [valid_set[0],valid_set[1]], \
            [test_set [0],test_set [1]]


def normalize(data, vdata=None, tdata=None):
    mu   = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    data = ( data - mu ) / std

    if vdata == None and tdata != None:
        tdata = (tdata - mu ) /std
        return data, tdata

    if vdata != None and tdata != None:
        vdata = (vdata - mu ) /std
        tdata = (tdata - mu ) /std
        return data, vdata, tdata
    return data


def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 

def share_input(x):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX))

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    #When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue

    return shared_x, T.cast(shared_y, 'int32')


'''Given tiles of raw data, this function will return training, validation, and test sets.
r_train - ratio of train set
r_valid - ratio of valid set
r_test  - ratio of test set'''
def gen_train_valid_test(raw_data, raw_target, r_train, r_valid, r_test):
    N = raw_data.shape[0]
    perms = np.random.permutation(N)
    raw_data   = raw_data[perms,:]
    raw_target = raw_target[perms]

    tot = float(r_train + r_valid + r_test)  #Denominator
    p_train = r_train / tot  #train data ratio
    p_valid = r_valid / tot  #valid data ratio
    p_test  = r_test / tot	 #test data ratio
    
    n_raw = raw_data.shape[0] #total number of data		
    n_train =int( math.floor(n_raw * p_train)) # number of train
    n_valid =int( math.floor(n_raw * p_valid)) # number of valid
    n_test  =int( math.floor(n_raw * p_test) ) # number of test

    
    train = raw_data[0:n_train, :]
    valid = raw_data[n_train:n_train+n_valid, :]
    test  = raw_data[n_train+n_valid: n_train+n_valid+n_test,:]
    
    train_target = raw_target[0:n_train]
    valid_target = raw_target[n_train:n_train+n_valid]
    test_target  = raw_target[n_train+n_valid: n_train+n_valid+n_test]
    
    print 'Among ', n_raw, 'raw data, we generated: '
    print train.shape[0], ' training data'
    print valid.shape[0], ' validation data'
    print test.shape[0],  ' test data\n'
    
    train_set = [train, train_target]
    valid_set = [valid, valid_target]
    test_set  = [test, test_target]
    return [train_set, valid_set, test_set]




if __name__ == '__main__':

    train_set, valid_set, test_set = load_dataset('./mnist.pkl.gz')
    print 123, train_set[1].shape

    train_set_x, train_set_y = shared_dataset(train_set);    
    test_set_x, test_set_y   = shared_dataset(test_set);    
    valid_set_x, valid_set_y = shared_dataset(valid_set); 

    print type(train_set_x)
    print dir(train_set_x)

    data  = train_set_x[2 * 500: 3 * 500]
    print data


