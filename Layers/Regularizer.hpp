#ifndef REGULARIZER_CPP
#define REGULARIZER_CPP

#include "Tensor.hpp"
#include "BaseLayer.hpp"

class Regularizer {
    public:
        Regularizer() = default;
        virtual ~Regularizer() = default;
        virtual double norm(const Tensor& input) = 0;
        virtual Tensor calculate_gradient(const Tensor& input) = 0;
};

#endif // REGULARIZER_CPP
