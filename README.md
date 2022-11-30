### PyFlow
#### A Deep Learning Framework

PyFlow is a deep learning framework implemented with C++ working on Python. It is a framework for building and training deep neural networks. It is designed to be modular and extensible, so that you can use just the parts you need. It is built from purely scratch, and it is not a wrapper of other deep learning frameworks. It is in development, and it is not finished yet. In the master branch, it is more stable than the dev branch. It will be planned to merge this project with the CUDA version of PyFlow in the future. (DL_Framework_CUDA also in my github). 

The framework consists of:

* Functional Tensor library
* Layers: Fully Connected
* Optimizers: SGD, Momentum, Adam
* Loss Functions: L2Loss, L1Loss, Cross Entropy Loss
* Activation Functions: Sigmoid, ReLU, Leaky ReLU
* Initializers: Constant, Uniform 
* Neural Network Class on Python to build and train neural networks
  * Save and Load Neural Network
* A binding of PyFlow to Python
* A data preprocessing notebook (can be generically used)
* (In dev branch) Schedulers: StepLR, ExponentialLR, MultiStepLR