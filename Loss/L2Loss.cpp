#include "L2Loss.hpp"

double L2Loss::forward(Tensor output, const Tensor& target) {
    last_input = output;
    return (output - target).power(2).sum();
}

Tensor L2Loss::backward(const Tensor& target) {
    return (last_input - target) * 2.0;
}