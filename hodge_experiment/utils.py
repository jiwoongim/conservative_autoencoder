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


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.rng_mrg as RNG_MRG


import numpy as np
import matplotlib.pyplot as plt

from numpy.lib import stride_tricks
import Image	# Read in an image.

def get_corrupted_input(rng, input, corruption_level, ntype='zeromask'):
    MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))
    #theano_rng = RandomStreams()
    if corruption_level == 0.0:
        return input

    if ntype=='zeromask':
        return MRG.binomial(size=input.shape, n=1, p=1-corruption_level,dtype=theano.config.floatX) * input
    elif ntype=='gaussian':
        return input + MRG.normal(size = input.shape, avg = 0.0,
                std = corruption_level, dtype = theano.config.floatX)
    elif ntype=='salt_pepper':

        # salt and pepper noise
        print 'DAE uses salt and pepper noise'
        a = MRG.binomial(size=input.shape, n=1,\
                p=1-corruption_level,dtype=theano.config.floatX)
        b = MRG.binomial(size=input.shape, n=1,\
                p=corruption_level,dtype=theano.config.floatX)

        c = T.eq(a,0) * b
        return input * a + c
