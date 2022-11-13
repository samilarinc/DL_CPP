#ifndef L1LOSS_HPP
#define L1LOSS_HPP

#include "Tensor.hpp"
#include "Loss.hpp"

class L1Loss : public Loss{
public:
    L1Loss() = default;
    ~L1Loss() = default;
    double forward(Tensor&, const Tensor&) override;
    Tensor backward(const Tensor&) override;

    Tensor last_input;
};

#endif // L1LOSS_HPP 