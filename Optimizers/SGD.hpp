#ifndef SGD_HPP
#define SGD_HPP

#include "BaseOptimizer.hpp"
#include "Tensor.hpp"
#include "Regularizer.hpp"
#include "L1_Regularizer.hpp"
#include "L2_Regularizer.hpp"

class SGD : public BaseOptimizer {
public:
    SGD(double lr, double momentum, string RegularizerType, double lambda);
    SGD(double lr, double momentum) : SGD(lr, momentum, "NULL", 0.0) {};
    SGD(double lr) : SGD(lr, 0.0, "NULL", 0.0) {};
    SGD(double lr, string RegularizerType, double lambda) : SGD(lr, 0.0, RegularizerType, lambda) {};
    ~SGD() = default;
    void calculate_update(Tensor&, Tensor) override;
    double getLearningRate() const;
    double getMomentum() const;

    double learning_rate;
    double momentum;
    Tensor velocity;
    Regularizer* regularizer = nullptr;
};

#endif