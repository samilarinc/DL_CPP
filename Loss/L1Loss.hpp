#ifndef L1LOSS_HPP  
#define L1LOSS_HPP

#include "Tensor.hpp"

class L1Loss {
public:
    L1Loss() = default;
    ~L1Loss() = default;
    double forward(Tensor, const Tensor&);
    Tensor backward(const Tensor&);

    Tensor last_input;
};

#endif // L1LOSS_HPP