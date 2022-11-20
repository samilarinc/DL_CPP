#ifndef ADAM_HPP
#define ADAM_HPP

#include "Tensor.hpp"
#include "BaseOptimizer.hpp"

class Adam : public BaseOptimizer
{
public:
    Adam(double, double, double);
    void calculate_update(Tensor&, Tensor) override;
    double getLearningRate() const;
    double getMu() const;
    double getRho() const;

    double lr;
    double mu;
    double rho;
    int k;
    Tensor v;
    Tensor r;
};

#endif