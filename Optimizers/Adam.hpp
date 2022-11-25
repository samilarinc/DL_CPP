#ifndef ADAM_HPP
#define ADAM_HPP

#include "Tensor.hpp"
#include "BaseOptimizer.hpp"
#include "Scheduler.hpp"

class Adam : public BaseOptimizer
{
public:
    Adam(double, double, double);
    void calculate_update(Tensor&, Tensor) override;
    double getLearningRate() const;
    double getMu() const;
    double getRho() const;
    void addScheduler(Scheduler) override;

    double lr;
    double mu;
    double rho;
    int k;
    Tensor v;
    Tensor r;
    Scheduler scheduler;
};

#endif