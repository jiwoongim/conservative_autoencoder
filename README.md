# conservative_autoencoder


Python (Theano) implementation of conservative auto-encoder code provided 
by Daniel Jiwoong Im and Mohamed Ishmael Diwan Belghazi.
The codes include experiments on hodge decomposition, in particular convservative components (for now),
and vector field deformations in 2D. For more information, see 

```bibtex
@article{Im2015cae,
    title={Conservativeness of untied auto-encoders},
    author={Im, Daniel Jiwoong and Belghanzi, Mohamed Ishmael Diwan and Memisevic, Roland},
    journal={http://arxiv.org/pdf/1506.07643.pdf},
    year={2015}
}
```

If you use this in your research, we kindly ask that you cite the above workshop paper


## Dependencies
Packages
* [numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)


## How to run
Entry code for one-bit flip and factored minimum probability flow for mnist data are 
```
    - /hodge_experiment/mnist_synthetic.py
    - /vector_field_deformation/mnist_train.py
```

