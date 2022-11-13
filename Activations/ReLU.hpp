#ifndef RELU_HPP
#define RELU_HPP

#include "BaseLayer.hpp"

class ReLU : public BaseLayer {
    public:
        ReLU();
        ~ReLU();
        Tensor forward(const Tensor&) override;
        Tensor backward(const Tensor&) override;

        Tensor last_input;
};

#endif