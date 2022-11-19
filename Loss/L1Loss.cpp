#include "L1Loss.hpp"

L1Loss::~L1Loss() {
    // if(last_input.data != nullptr) {
    //     free(last_input.data);
    // }
}

double L1Loss::forward(Tensor& output, const Tensor& target) {
    last_input = output;
    return (output - target).abs().sum() / output.getBatchsize();
}

Tensor L1Loss::backward(const Tensor& target) {
    return (last_input - target).sign() / last_input.getBatchsize();
}