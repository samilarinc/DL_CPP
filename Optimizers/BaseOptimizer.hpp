#ifndef BASEOPTIMIZER_HPP
#define BASEOPTIMIZER_HPP

#include "Tensor.hpp"
#include "Scheduler.hpp"

class BaseOptimizer {
public:
    BaseOptimizer() = default;
    virtual ~BaseOptimizer() = default;
    virtual void calculate_update(Tensor&, Tensor) = 0;
    virtual void addScheduler(Scheduler) = 0;
};

#endif