#include "Softmax.hpp"

Softmax::Softmax() = default;
Softmax::~Softmax() = default;

Tensor Softmax::forward(const Tensor& input) {
    this->last_input = input.copy();
    Tensor output = Tensor(input.getBatchsize(), input.getRows(), input.getCols());
    for(int i = 0; i < input.getBatchsize(); i++) {
        double sum = 0.0;
        for(int j = 0; j < input.getRows(); j++) {
            sum += exp(input(i, j, 0));
        }
        for(int j = 0; j < input.getRows(); j++) {
            output(i, j, 0) = exp(input(i, j, 0)) / sum;
        }
    }
    return output;
}