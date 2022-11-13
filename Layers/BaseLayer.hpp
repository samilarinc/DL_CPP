#ifndef BASELAYER_HPP
#define BASELAYER_HPP

#include "Tensor.hpp"

class BaseLayer {
public:
    BaseLayer() = default;
    virtual ~BaseLayer() = default;
    virtual Tensor forward(const Tensor&) = 0;
    virtual Tensor backward(const Tensor&) = 0;
    bool trainable = false;
};
#endif