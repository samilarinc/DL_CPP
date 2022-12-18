#include "Dropout.hpp"

Dropout::Dropout(double p) : p(p) {}

Tensor Dropout::forward(const Tensor& input) {
    if (p == 0.0 || p == 1.0) {
        return input;
    }
    if (is_training){
        srand(time(0));
        mask = Tensor(input.getRows(), input.getCols(), input.getBatchsize(), 1.0);
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                for (int k = 0; k < input.getBatchsize(); k++) {
                    if (rand() % 100 < p * 100) {
                        mask(i, j, k) = 0.0;
                    }
                }
            }
        }
        Tensor temp = input.copy();
        return temp * mask;
    }
    return input;
}

Tensor Dropout::backward(const Tensor& input) {
    if (p == 0.0 || p == 1.0) {
        return input;
    }
    Tensor temp = input.copy();
    return temp * mask;
}

void Dropout::save(string path, int num) {
    ofstream file(path + "/Dropout" + to_string(num) + ".Dropout");
    file << p << endl;
    file.close();
}

void Dropout::load(string path, int num) {
    ifstream file(path + "/" + to_string(num) + ".Dropout");
    file >> p;
    file.close();
}

void Dropout::set_training(bool is_training) {
    this->is_training = is_training;
}