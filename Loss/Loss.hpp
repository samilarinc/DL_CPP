#ifndef LOSS_HPP
#define LOSS_HPP

#include "Tensor.hpp"

class Loss{
public:
    Loss() = default;
    virtual ~Loss() = default;
    virtual double forward(Tensor&, const Tensor&) = 0;
    virtual Tensor backward(const Tensor&) = 0;

    Tensor last_input;
};

#endif // LOSS_HPP