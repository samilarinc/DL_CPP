#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include "Tensor.hpp"
#include "Loss.hpp"

class CrossEntropyLoss : public Loss{
public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss();
    double forward(Tensor&, const Tensor&) override;
    Tensor backward(const Tensor&) override;

    Tensor last_input;
};

#endif // CROSS_ENTROPY_LOSS_HPP