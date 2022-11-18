#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "BaseLayer.hpp"
#include "Tensor.hpp"

class Softmax : public BaseLayer {
public:
    Softmax();
    ~Softmax();
    void forward(const Tensor&) override;
    void backward(const Tensor&) override;

    Tensor last_input;    
};