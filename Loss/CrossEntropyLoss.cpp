#include "CrossEntropyLoss.hpp"
#include <math.h>

#define EPSILON 1e-9

CrossEntropyLoss::~CrossEntropyLoss() {
    if(this->last_input.data != nullptr) {
        delete this->last_input.data;
    }
}

double CrossEntropyLoss::forward(Tensor& input, const Tensor& target){
    last_input = input.copy();
    double loss = 0;
    for(int i = 0; i < input.getBatchsize(); i++){
        for(int j = 0; j < input.getRows(); j++){
            for(int k = 0; k < input.getCols(); k++){
                loss += target(i, j, k) * log(input(i, j, k) + EPSILON);
            }
        }
    }
    return loss;
}

Tensor CrossEntropyLoss::backward(const Tensor &target){
    Tensor output(last_input.getRows(), last_input.getCols(), last_input.getBatchsize());
    for(int i = 0; i < last_input.getBatchsize(); i++){
        for(int j = 0; j < last_input.getRows(); j++){
            for(int k = 0; k < last_input.getCols(); k++){
                output(i, j, k) = target(i, j, k) / (last_input(i, j, k) + EPSILON);
            }
        }
    }
    return output;
}