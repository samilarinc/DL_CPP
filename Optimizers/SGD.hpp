#ifndef SGD_HPP
#define SGD_HPP

#include "BaseOptimizer.hpp"
#include "Tensor.hpp"

class SGD : public BaseOptimizer {
public:
    SGD(double);
    ~SGD() = default;
    void calculate_update(Tensor&, Tensor) override;
    double getLearningRate() const;
    // SGD* clone() const;
private:
    double learning_rate;
};

#endif