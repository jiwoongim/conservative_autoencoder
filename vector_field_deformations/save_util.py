import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy 
import numpy as np


import os
import sys
import cPickle



def save_the_weight(x,fname):
    f = file(fname+'.save', 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


