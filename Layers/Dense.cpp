#include "Dense.hpp"
#include "Tensor.hpp"
#include "SGD.hpp"
#include "BaseOptimizer.hpp"
#include <string>

Dense::Dense(int input_size, int output_size, double lr, string opt, double mu_mentum, double rho) {
    this->input_size = input_size;
    this->output_size = output_size;
    weights = Tensor(1, output_size, input_size, 1);
    bias = Tensor(1, output_size, 1, 0.1);
    if(opt == "SGD") {
        printf("SGD");
        this->optimizer = new SGD(lr);
        this->bias_optimizer = new SGD(lr);
    }
}

Dense::~Dense(){
    if(last_input.data != nullptr) {
        free(last_input.data);
    }
    if(weights.data != nullptr) {
        free(weights.data);
    }
    if(bias.data != nullptr) {
        free(bias.data);
    }
    if(gradient_weights.data != nullptr) {
        free(gradient_weights.data);
    }
    if(gradient_bias.data != nullptr) {
        free(gradient_bias.data);
    }
    if(optimizer != nullptr) {
        delete optimizer;
    }
    if(bias_optimizer != nullptr) {
        delete bias_optimizer;
    }
}

Tensor Dense::getWeights() const {
    return weights;
}

Tensor Dense::getBias() const {
    return bias;
}

Tensor Dense::forward(const Tensor& input) { // batch size should be 1! Data should be flattened!
    Tensor output = Tensor(input.getBatchsize(), output_size, 1);
    last_input = Tensor(input);
    for(int i = 0; i < input.getBatchsize(); i++) {
        for(int j = 0; j < output_size; j++) {
            for(int k = 0; k < input_size; k++) {
                output(i, j, 0) += input(i, k, 0) * weights(0, j, k);
            }
        output(i, j, 0) += bias(0, j, 0);
        }
    }
    return output;
}

Tensor Dense::backward(const Tensor& error) {
    int batch_size = error.getBatchsize();
    Tensor dx(batch_size, input_size, 1);
    Tensor dw(1, output_size, input_size);
    Tensor db(1, output_size, 1);
    
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < output_size; j++) {
            for(int k = 0; k < input_size; k++) {
                dw(0, j, k) += error(i, j, 0) * last_input(i, k, 0);
                dx(i, k, 0) += error(i, j, 0) * weights(0, j, k);
            }
            db(0, j, 0) += error(i, j, 0);
        }
    }

    if(optimizer != nullptr) {
        optimizer->calculate_update(weights, dw);
    }
    if(bias_optimizer != nullptr) {
        bias_optimizer->calculate_update(bias, db);
    }

    gradient_bias = db;
    gradient_weights = dw;
    return dx;
}