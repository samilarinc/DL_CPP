#ifndef L1_REGULARIZER_CPP
#define L1_REGULARIZER_CPP

#include "Tensor.hpp"
#include "Regularizer.hpp"

class L1Regularizer: public Regularizer {
    public:
        L1Regularizer(double lambda);
        ~L1Regularizer();
        double norm(const Tensor& input) override;
        Tensor calculate_gradient(const Tensor& input) override;
    private:
        double lambda;
};

#endif // L1_REGULARIZER_CPP