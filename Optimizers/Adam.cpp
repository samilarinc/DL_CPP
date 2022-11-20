#include "Adam.hpp"
#include <math.h>

#define EPSILON 1e-8

Adam::Adam(double lr, double mu, double rho) : lr(lr), mu(mu), rho(rho), k(1) {}

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