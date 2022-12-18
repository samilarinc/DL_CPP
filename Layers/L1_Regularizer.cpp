#include "L1_Regularizer.hpp"

L1Regularizer::L1Regularizer(double lambda) {
    this->lambda = lambda;
}

L1Regularizer::~L1Regularizer() = default;

double L1Regularizer::norm(const Tensor& input) {
    return lambda * input.abs().sum();
}

Tensor L1Regularizer::calculate_gradient(const Tensor& input) {
    Tensor output = input.copy();
    return output * this->lambda;
}