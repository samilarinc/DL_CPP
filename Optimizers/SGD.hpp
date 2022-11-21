#ifndef SGD_HPP
#define SGD_HPP

#include "BaseOptimizer.hpp"
#include "Tensor.hpp"

class SGD : public BaseOptimizer {
public:
    SGD(double);
    SGD(double, double);
    ~SGD() = default;
    void calculate_update(Tensor&, Tensor) override;
    double getLearningRate() const;
    double getMomentum() const;

    double learning_rate;
    double momentum;
    Tensor velocity;
};

#endif