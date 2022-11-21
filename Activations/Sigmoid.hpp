#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "BaseLayer.hpp"
#include "Tensor.hpp"
#include <math.h>
#include <fstream>

class Sigmoid : public BaseLayer {
    public:
        Sigmoid() = default;
        ~Sigmoid() = default;
        Tensor forward(const Tensor&) override;
        Tensor backward(const Tensor&) override;
        void save(string, int) override;
        void load(string, int) override;
        Tensor last_input;
};

#endif // SIGMOID_HPP