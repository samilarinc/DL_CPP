#include "L1Loss.hpp"

double L1Loss::forward(Tensor output, const Tensor& target) {
    last_input = output;
    return (output - target).abs().sum();
}

Tensor L1Loss::backward(const Tensor& target) {
    return (last_input - target).sign();
}