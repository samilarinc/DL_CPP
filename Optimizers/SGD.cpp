#include "SGD.hpp"

SGD::SGD(double learning_rate) {
    this->learning_rate = learning_rate;
}

void SGD::calculate_update(Tensor& weights, Tensor gradient_weights) {
    weights = weights - (gradient_weights * this->learning_rate);
}

double SGD::getLearningRate() const {
    return this->learning_rate;
}