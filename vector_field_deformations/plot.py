import math
import theano
import theano.tensor as T
import numpy as np
import pickle
import pylab
from tile_img import *
from pylab import *

def plot_two(attempts, y1, y2, l1='Number of Labels', l2='Accuracy', n1=None,n2=None):
    
    p1 = pylab.plot(attempts, y1, 'b.-', label=n1)
    p2 = pylab.plot(attempts, y2, 'rx-', label=n2)
    pylab.legend()
    pylab.xlabel(l1)
    pylab.ylabel(l2)

def plot_three(attempts, y1, y2, y3, l1='Number of Labels', l2='Accuracy', \
        n1=None,n2=None,n3=None):
    
    pylab.figure()
    p1= pylab.plot(attempts, y1, 'g.-', label=n1)
    p2= pylab.plot(attempts, y2, 'rx-', label=n2)
    p3= pylab.plot(attempts, y3, 'b*-', label=n3)
    pylab.legend()
    pylab.xlabel(l1, fontsize=18)
    pylab.ylabel(l2, fontsize=18)
    pylab.show()   

def plot_four(attempts, y1, y2, y3, y4, l1='Number of Labels', l2='Accuracy', \
        n1=None,n2=None,n3=None,n4=None):

    pylab.figure()   
    p1= pylab.plot(attempts, y1, 'k--', label=n1)
    p2= pylab.plot(attempts, y2, 'g<-', label=n2)
    p3= pylab.plot(attempts, y3, 'b.-', label=n3)
    p4= pylab.plot(attempts, y4, 'r*-', label=n4)
    pylab.legend()
    pylab.xlabel(l1, fontsize=18)
    pylab.ylabel(l2, fontsize=18)
    #pylab.show()   

def plot_five(attempts, y1, y2, y3, y4, y5, l1='Number of Labels', l2='Accuracy', \
        n1=None,n2=None,n3=None,n4=None,n5=None,):
    
    p1= pylab.plot(attempts, y1, 'b.-', label=n1)
    p2= pylab.plot(attempts, y2, 'g4-', label=n2)
    p3= pylab.plot(attempts, y3, 'rx-', label=n3)
    p4= pylab.plot(attempts, y4, 'k--', label=n4)
    p5= pylab.plot(attempts, y5, 'md-', label=n5)
    pylab.legend(loc='lower left')
    pylab.xlabel(l1, fontsize=18)
    pylab.ylabel(l2, fontsize=18)
    pylab.show()   

def plot_six(attempts, y1, y2, y3, y4, y5,y6, l1='Number of Labels', l2='Accuracy', \
        n1=None,n2=None,n3=None,n4=None,n5=None, n6=None):
    
    p1= pylab.plot(attempts, y1, 'b.-', label=n1)
    p2= pylab.plot(attempts, y2, 'g4-', label=n2)
    p3= pylab.plot(attempts, y3, 'rx-', label=n3)
    p4= pylab.plot(attempts, y4, 'k--', label=n4)
    p5= pylab.plot(attempts, y5, 'md-', label=n5)
    p6= pylab.plot(attempts, y6, 'c>-', label=n6)
    pylab.legend()
    pylab.legend(loc='lower left')
    pylab.xlabel(l1, fontsize=12)
    pylab.ylabel(l2, fontsize=12)
    pylab.show()   

def plot_seven(attempts, y1, y2, y3, y4, y5, y6, y7, l1='Number of Labels', l2='Accuracy', \
        n1=None,n2=None,n3=None,n4=None,n5=None, n6=None, n7=None):
    
    p1= pylab.plot(attempts, y1, 'b.-', label=n1)
    p2= pylab.plot(attempts, y2, 'g4-', label=n2)
    p3= pylab.plot(attempts, y3, 'rx-', label=n3)
    p4= pylab.plot(attempts, y4, 'k--', label=n4)
    p5= pylab.plot(attempts, y5, 'md-', label=n5)
    p6= pylab.plot(attempts, y6, 'c>-', label=n6)
    p7= pylab.plot(attempts, y7, 'p:' , label=n7)
    pylab.legend()
    pylab.legend(loc='lower left')
    pylab.xlabel(l1, fontsize=18)
    pylab.ylabel(l2, fontsize=18)
    pylab.show()   

def plot_eight(attempts, y1, y2, y3, y4, y5, y6, y7, y8, l1='Number of Labels', l2='Accuracy', \
        n1=None,n2=None,n3=None,n4=None,n5=None, n6=None, n7=None, n8=None):
    
    p1= pylab.plot(attempts, y1, 'b.-', label=n1)
    p2= pylab.plot(attempts, y2, 'g4-', label=n2)
    p3= pylab.plot(attempts, y3, 'rx-', label=n3)
    p4= pylab.plot(attempts, y4, 'k--', label=n4)
    p5= pylab.plot(attempts, y5, 'md-', label=n5)
    p6= pylab.plot(attempts, y6, 'c>-', label=n6)
    p7= pylab.plot(attempts, y7, 'p:' , label=n7)
    p8= pylab.plot(attempts, y8, 'yo-', label=n8)
    pylab.legend(loc='lower left')
    pylab.xlabel(l1)
    pylab.ylabel(l2)
    pylab.show()   

def plot_nine(attempts, y1, y2, y3, y4, y5, y6, y7, y8, y9, l1='Number of Labels', l2='Accuracy', \
        n1=None,n2=None,n3=None,n4=None,n5=None, n6=None, n7=None, n8=None, n9=None):
    
    p1= pylab.plot(attempts, y1, 'b.-', label=n1)
    p2= pylab.plot(attempts, y2, 'g4-', label=n2)
    p3= pylab.plot(attempts, y3, 'rx-', label=n3)
    p4= pylab.plot(attempts, y4, 'k--', label=n4)
    p5= pylab.plot(attempts, y5, 'md-', label=n5)
    p6= pylab.plot(attempts, y6, 'c>-', label=n6)
    p7= pylab.plot(attempts, y7, 'p:' , label=n7)
    p8= pylab.plot(attempts, y8, 'yo-', label=n8)
    p9= pylab.plot(attempts, y9, 'go-', label=n9)
    pylab.legend(loc='lower left')
    pylab.xlabel(l1, fontsize=18)
    pylab.ylabel(l2, fontsize=18)
    pylab.show()   



def plot_cost(epochs, x, y, l1='Training data', l2='Validation data'):
    pylab.plot(epochs, x, '-b', label=l1)
    pylab.plot(epochs, y, '-r', label=l2)
    pylab.legend(loc='upper right')
    pylab.xlabel("Epoch")
    pylab.ylabel("Cross entropy")
    pylab.show()


def plot_attempts(x, y, xl,yl, label):
    pylab.plot(x, y, '-b', label=label)
    pylab.xlabel(xl)
    pylab.ylabel(yl)
    pylab.show()


def display_weight(W, fig_label, fig_i, img_shape=(28,28), tile_shape=(10,10), fname=None):

        #display_dataset(W, (28,28), (10,10),scale_rows_to_unit_interval=False, binary=False)
        tmp = tile_raster_images(X=W, img_shape=img_shape, tile_shape=tile_shape, \
                            tile_spacing=(10,10), scale_rows_to_unit_interval=True)
    

        if fname != None:
            im = PIL.Image.fromarray(tmp)
            im.save(fname+'.png')
            
        else:
            fig = plt.figure(fig_i)
            plt.title(fig_label)
            plt.imshow(tmp,cmap='gray')


def display_input_recon(input, recon, patch_sz=(28,28), tile_shape=(10,10), fname=None):

    D = input.shape[0]
    #input = np.round(input ).astype('uint8')
    #recon = np.round(recon ).astype('uint8')

    fname1 = None; fname2 = None
    if fname != None:
        fname1 = fname + '_org'
        fname2 = fname + '_rec'

    display_dataset(input, patch_sz, tile_shape, scale_rows_to_unit_interval=False, \
                                binary=False,i=432, fname=fname1)
    display_dataset(recon, patch_sz, tile_shape, scale_rows_to_unit_interval=False, \
                                binary=False,i=234, fname=fname2)


def display_activation(num_hid, num_vis, W, fpath=None, num_view=100):

    #assert W.get_value().shape == [num_vis, num_hid], 'Weight Dimensions need to be matched'
    activation_vis = T.zeros((num_hid, num_vis))

    for hid in xrange(num_view):
        activation_vis = T.set_subtensor(activation_vis[hid,:], W[:,hid] / T.sqrt(T.sum(W[:,hid])**2))

    display_weight(activation_vis[0:num_view].eval(), 'visualizing_activation', 33, tile_shape=(10,10), fname=fpath+'activation')
    pass


def display_boxplot():
    spread=rand(50)*100
    center = ones(25)*50
    filer_high = rand(10) * 100 - 100
    filer_low = rand(10)*-100
    data = concatenate((spread, center, filer_high, filer_low),0)

    #basic plot
    boxplot(data,1)
    show()


if __name__ == '__main__':
    display_boxplot()
    
