#include "SGD.hpp"

SGD::SGD(double learning_rate, double momentum, string RegularizerType, double lambda) {
    this->learning_rate = learning_rate;
    this->momentum = momentum;
    if(RegularizerType == "L1") {
        this->regularizer = new L1Regularizer(lambda);
    }
    else if(RegularizerType == "L2") {
        this->regularizer = new L2Regularizer(lambda);
    }
    else {
        this->regularizer = nullptr;
    }
}

void SGD::calculate_update(Tensor& weights, Tensor gradient_weights) {
    if(this->momentum != 0.0) {
        if(this->velocity.getRows() == 0) { // Initialize velocity first time
            this->velocity = Tensor(gradient_weights); 
        }
        this->velocity = this->velocity * this->momentum + gradient_weights;
        weights = weights - this->velocity * this->learning_rate;
    }
    else {
        weights = weights - gradient_weights * this->learning_rate;
    }
    if (this->regularizer != nullptr) {
        this->regularizer->calculate_gradient(weights);
    }
}

double SGD::getLearningRate() const {
    return this->learning_rate;
}

double SGD::getMomentum() const {
    return this->momentum;
}