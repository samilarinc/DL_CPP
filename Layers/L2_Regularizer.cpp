#include "L2_Regularizer.hpp"

L2Regularizer::L2Regularizer(double lambda) {
    this->lambda = lambda;
}

L2Regularizer::~L2Regularizer() = default;

double L2Regularizer::norm(const Tensor& input) {
    Tensor output = input.copy();
    return lambda * output.power(2).sum();
}

Tensor L2Regularizer::calculate_gradient(const Tensor& input) {
    return input.sign() * lambda;
}