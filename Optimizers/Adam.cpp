#include "Adam.hpp"
#include <math.h>

#define EPSILON 1e-8

Adam::Adam(double lr, double mu, double rho, string RegularizerType, double lambda) {
    this->lr = lr;
    this->mu = mu;
    this->rho = rho;
    this->k = 1;
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

void Adam::calculate_update(Tensor& weights, Tensor grad) {
    if(v.getRows() == 0)
    {
        v = Tensor(weights.getBatchsize(), weights.getRows(), weights.getCols(), 0.0);
        r = Tensor(weights.getBatchsize(), weights.getRows(), weights.getCols(), 0.0);
    }
    v = v * mu + grad * (1 - mu);
    r = r * rho + grad.power(2) * (1 - rho);
    Tensor v_hat = v / (1 - pow(mu, k));
    Tensor r_hat = r / (1 - pow(rho, k));
    k++;
    weights = weights - ((v_hat * lr) / (r_hat.power(0.5) + EPSILON));
    if (this->regularizer != nullptr) {
        this->regularizer->calculate_gradient(weights);
    }
}

double Adam::getLearningRate() const {
    return lr;
}

double Adam::getMu() const {
    return mu;
}

double Adam::getRho() const {
    return rho;
}