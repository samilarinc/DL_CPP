#ifndef L2LOSS_HPP
#define L2LOSS_HPP

#include "Tensor.hpp"
#include "Loss.hpp"

class L2Loss : public Loss{
public:
    L2Loss() = default;
    ~L2Loss() = default;
    double forward(Tensor&, const Tensor&) override;
    Tensor backward(const Tensor&) override;

    Tensor last_input;
};

#endif // L2LOSS_HPP 