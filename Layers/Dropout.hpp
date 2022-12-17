#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include <stdlib.h>
#include <time.h>

#include <fstream>
#include <sstream>

#include "Tensor.hpp"
#include "BaseLayer.hpp"


class Dropout : public BaseLayer{
public:
    Dropout(double);
    ~Dropout() = default;
    Tensor forward(const Tensor&) override;
    Tensor backward(const Tensor&) override;
    void save(string, int) override;
    void load(string, int) override;
    double set_training(bool);

private:
    double p;
    Tensor mask;
    bool is_training = true;
};

#endif // DROPOUT_HPP