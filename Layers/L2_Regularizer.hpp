#ifndef L2_REGULARIZER_CPP
#define L2_REGULARIZER_CPP

#include "Tensor.hpp"
#include "Regularizer.hpp"

class L2Regularizer: public Regularizer {
    public:
        L2Regularizer(double lambda);
        ~L2Regularizer();
        double norm(const Tensor& input) override;
        Tensor calculate_gradient(const Tensor& input) override;
    private:
        double lambda;
};

#endif // L2_REGULARIZER_CPP