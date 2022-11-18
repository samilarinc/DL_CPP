#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Tensor.hpp"
#include "BaseLayer.hpp"

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void append_layer(BaseLayer* layer);
    void forward(Tensor* input);
    void backward(Tensor* target);
    void update(double learning_rate);
    bool get_testing_phase();

    Tensor* getOutput();

    bool testing_phase = false;
}