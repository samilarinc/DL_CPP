#ifndef L2LOSS_HPP
#define L2LOSS_HPP

#include "Tensor.hpp"

class L2Loss {
public:
    L2Loss() = default;
    ~L2Loss() = default;
    double forward(Tensor, const Tensor&);
    Tensor backward(const Tensor&);

    Tensor last_input;
};

#endif // L2LOSS_HPP 