#ifndef RELU_HPP
#define RELU_HPP

#include "BaseLayer.hpp"
#include "Tensor.hpp"
#include <fstream>

class ReLU : public BaseLayer {
    public:
        ReLU() = default;
        ~ReLU() = default;
        Tensor forward(const Tensor&) override;
        Tensor backward(const Tensor&) override;
        void save(string, int) override;
        void load(string, int) override;
        Tensor last_input;
};

#endif