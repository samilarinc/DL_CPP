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


### **Future Updates**

Here are a few suggestions for optimizing this code:

Use std::array instead of std::vector for storing the tensor data. std::array is a fixed-size container, so it will use less memory and be more efficient than std::vector, which is a dynamically-allocated container.

Consider using std::unique_ptr instead of a raw pointer for storing the tensor data. std::unique_ptr is a smart pointer that manages the memory for the tensor data and automatically deletes the data when the Tensor object is destroyed. This will prevent memory leaks and make the code easier to maintain.

Use std::move in the copy and move constructors and assignment operators to move the data from the source tensor to the new Tensor object instead of copying it. This will improve the performance of these operations, especially for large tensors.

Use std::get<> to access the elements of the tuple that stores the tensor shape instead of accessing them directly. This will make the code more readable and easier to maintain.

Consider using C++11 auto and range-based for loops in the methods that iterate over the tensor data. This will make the code more concise and easier to read.

Use C++11 nullptr instead of NULL in the default constructor. nullptr is the preferred way to represent a null pointer in C++11 and later.

Use std::swap in the swap method instead of manually swapping the values of the tensor data and shape. This will make the code more efficient and easier to read.

Use std::make_unique to create the tensor data in the constructors instead of using new and delete. This will make the code easier to read and maintain, and will prevent memory leaks.

Use std::fill to initialize the tensor data with a constant value instead of using a loop. This will make the code more efficient and easier to read.

Use std::uniform_real_distribution instead of std::random_device and std::mt19937 to generate random numbers for initializing the tensor with random values. This will make the code more concise and easier to read.