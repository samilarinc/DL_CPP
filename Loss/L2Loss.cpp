#include "L2Loss.hpp"

L2Loss::~L2Loss() {
    if(last_input.data != nullptr) {
        free(last_input.data);
    }
}

double L2Loss::forward(Tensor& output, const Tensor& target) {
    last_input = output.copy();
    return (output - target).power(2).sum();
}

Tensor L2Loss::backward(const Tensor& target) {
    return (last_input - target) * 2.0;
}