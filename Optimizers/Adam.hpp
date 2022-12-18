#ifndef ADAM_HPP
#define ADAM_HPP

#include "Tensor.hpp"
#include "BaseOptimizer.hpp"
#include "Regularizer.hpp"
#include "L1_Regularizer.hpp"
#include "L2_Regularizer.hpp"

class Adam : public BaseOptimizer
{
public:
    Adam(double lr, double mu, double rho, string RegularizerType, double lambda);
    Adam(double lr, double mu, double rho) : Adam(lr, mu, rho, "NULL", 0) {};
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
    Regularizer* regularizer = nullptr;
};

#endif