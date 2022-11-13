#include "SGD.hpp"

SGD::SGD(double learning_rate) {
    this->learning_rate = learning_rate;
    this->momentum = 0.0;
    this->velocity = Tensor();
}

SGD::SGD(double learning_rate, double momentum) {
    this->learning_rate = learning_rate;
    this->momentum = momentum;
    this->velocity = Tensor(0, 0, 0);
}

void SGD::calculate_update(Tensor& weights, Tensor gradient_weights) {
    weights = weights - (gradient_weights * this->learning_rate);
    if(this->momentum != 0.0) {
        if(this->velocity.getRows() == 0) { // Initialize velocity first time
            this->velocity = Tensor(gradient_weights.getBatchsize(), gradient_weights.getRows(), gradient_weights.getCols()); 
        }
        this->velocity = this->velocity * this->momentum + gradient_weights * this->learning_rate;
        weights = weights - this->velocity;
    }
}

double SGD::getLearningRate() const {
    return this->learning_rate;
}