#ifndef SGD_HPP
#define SGD_HPP

#include "BaseOptimizer.hpp"
#include "Tensor.hpp"
#include "Scheduler.hpp"
#include <vector>
#include <string>

class SGD : public BaseOptimizer {
public:
    SGD(double);
    SGD(double, double);
    ~SGD() = default;
    void calculate_update(Tensor&, Tensor) override;
    double getLearningRate() const;
    double getMomentum() const;
    void addScheduler(Scheduler) override;

    double learning_rate;
    double momentum;
    Tensor velocity;
    Scheduler scheduler;
    int current_epoch = 0;
};

#endif