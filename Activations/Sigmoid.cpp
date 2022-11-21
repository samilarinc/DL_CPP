#include "Sigmoid.hpp"

Tensor Sigmoid::forward(const Tensor& input) {
    Tensor output(input.getBatchsize(), input.getRows(), input.getCols());
    for (int i = 0; i < input.getBatchsize(); i++) {
        for (int j = 0; j < input.getRows(); j++) {
            for (int k = 0; k < input.getCols(); k++) {
                output(i, j, k) = 1 / (1 + exp(-input(i, j, k)));
            }
        }
    }
    last_input = output.copy();
    return output;
}

Tensor Sigmoid::backward(const Tensor& error) {
    Tensor output(error.getBatchsize(), error.getRows(), error.getCols());
    for (int i = 0; i < error.getBatchsize(); i++) {
        for (int j = 0; j < error.getRows(); j++) {
            for (int k = 0; k < error.getCols(); k++) {
                output(i, j, k) = error(i, j, k) * last_input(i, j, k) * (1 - last_input(i, j, k));
            }
        }
    }
    return output;
}

void Sigmoid::save(string path, int index) {
    ofstream file(path + "/" + to_string(index) + ".Sigmoid");
    file.close();
}

void Sigmoid::load(string path, int index) {
    ifstream file(path + "/" + to_string(index) + ".Sigmoid");
    file.close();
}