#ifndef DENSE_HPP
#define DENSE_HPP

#include "Tensor.hpp"
#include "BaseLayer.hpp"
#include "BaseOptimizer.hpp"
#include "SGD.hpp"
#include "Adam.hpp"
#include "Scheduler.hpp"
#include <string>

class Dense : public BaseLayer {
public:
    Dense() = default;
    Dense(int input_size, int output_size, double lr, string opt, double momentum, double mu, double rho); // Constructor
    Dense(int input_size, int output_size) : Dense(input_size, output_size, 0, "NULL", 0, 0, 0) {}; // No optimizer
    Dense(int input_size, int output_size, double lr, string opt) : Dense(input_size, output_size, lr, opt, 0, 0, 0) {}; // SGD
    Dense(int input_size, int output_size, double lr, string opt, double momentum) : Dense(input_size, output_size, lr, opt, momentum, 0, 0) {}; // SGD with momentum
    Dense(int input_size, int output_size, double lr, string opt, double mu, double rho) : Dense(input_size, output_size, lr, opt, 0, mu, rho) {}; // Adam
    ~Dense();
    Tensor getWeights() const;
    Tensor getBias() const;
    Tensor forward(const Tensor&) override;
    Tensor backward(const Tensor&) override;
    void save(string, int) override;
    void load(string, int) override;
    bool trainable = true;
    void setScheduler(Scheduler);
    // void setOptimizer(BaseOptimizer*);
    // BaseOptimizer* getOptimizer() const;
    Tensor gradient_weights;
    Tensor gradient_bias;
// private:
    int input_size;
    int output_size;
    Tensor weights;
    Tensor bias;
    Tensor last_input;
    string optimizer_type;
    BaseOptimizer* optimizer = nullptr;
    BaseOptimizer* bias_optimizer = nullptr;
    int current_epoch = 0;
};
#endif // DENSE_HPP