#include "ReLU.hpp"

Tensor ReLU::forward(const Tensor& input) {
    Tensor output = Tensor(input.getBatchsize(), input.getRows(), input.getCols());
    last_input = Tensor(input);
    for(int i = 0; i < input.getBatchsize(); i++) {
        for(int j = 0; j < input.getRows(); j++) {
            for(int k = 0; k < input.getCols(); k++) {
                output(i, j, k) = input(i, j, k) > 0.0 ? input(i, j, k) : 0.0;
            }
        }
    }
    return output;
}

Tensor ReLU::backward(const Tensor& error) {
    Tensor output = Tensor(last_input.getBatchsize(), last_input.getRows(), last_input.getCols());
    for(int i = 0; i < last_input.getBatchsize(); i++) {
        for(int j = 0; j < last_input.getRows(); j++) {
            for(int k = 0; k < last_input.getCols(); k++) {
                output(i, j, k) = last_input(i, j, k) > 0.0 ? error(i, j, k) : 0.0;
            }
        }
    }
    return output;
}

void ReLU::save(string path, int num) {
    ofstream file(path + "/" + to_string(num) + ".ReLU");
    file.close();
}

void ReLU::load(string path, int num) {
    ifstream file(path + "/" + to_string(num) + ".ReLU");
    file.close();
}