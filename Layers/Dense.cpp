#include "Dense.hpp"

Dense::Dense(int input_size, int output_size, double lr, string opt, double momentum, double mu, double rho, string initializer) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->optimizer_type = opt;
    if (strcmp(initializer.c_str(), "constant") == 0) {
        weights = Tensor(1, output_size, input_size, 0.1, 1.0, initializer);
        bias = Tensor(1, output_size, 1, 0.0, 1.0, initializer);
    } else {
        weights = Tensor(1, output_size, input_size, 0.0, 1.0, initializer);
        bias = Tensor(1, output_size, 1, 0.0, 1.0, initializer);
    }
    if(opt == "SGD") {
        // printf("SGD Optimizer with learning rate %f and momentum %f\n\n", lr, momentum);
        this->optimizer = new SGD(lr, momentum);
        this->bias_optimizer = new SGD(lr, momentum);
    }
    else if(opt == "Adam") {
        // printf("Adam Optimizer with learning rate %f, mu %f and rho %f\n\n", lr, mu, rho);
        this->optimizer = new Adam(lr, mu, rho);
        this->bias_optimizer = new Adam(lr, mu, rho);
    }
    else {
        // printf("No optimizer specified, using SGD with learning rate %f and momentum %f\n\n", lr, momentum);
        this->optimizer = nullptr;
        this->bias_optimizer = nullptr;
    }
}

Dense::~Dense() = default;

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
    dw = dw / batch_size;
    db = db / batch_size;

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

void Dense::save(string path, int num) {
    ofstream file(path + "/" + to_string(num) + ".FC");
    if(file.is_open()) {
        file << to_string(input_size) + " " + to_string(output_size);
        file << endl;
        file << optimizer_type;
        file << endl;
        if(optimizer_type == "SGD") {
            SGD* sgd = (SGD *)optimizer;
            file << to_string(sgd->getLearningRate()) + " " + to_string(sgd->getMomentum());
            file << endl;
        }
        else if(optimizer_type == "Adam") {
            Adam *adam = (Adam *)optimizer;
            file << to_string(adam->getLearningRate()) + " " + to_string(adam->getMu()) + " " + to_string(adam->getRho());
            file << endl;
        }
        for(int i = 0; i < output_size; i++) {
            for(int j = 0; j < input_size; j++) {
                file << weights(0, i, j) << " ";
            }
        }
        file << endl;
        for(int i = 0; i < output_size; i++) {
            file << bias(0, i, 0) << " ";
        }
    }
    file.close();
}

void Dense::load(string path, int num) {
    ifstream file(path + "/" + to_string(num) + ".FC");
    if(file.is_open()) {
        string line;
        getline(file, line);
        istringstream iss(line);
        iss >> input_size >> output_size;
        getline(file, line);
        istringstream iss2(line);
        iss2 >> optimizer_type;
        if(optimizer_type == "SGD") {
            double lr, momentum;
            getline(file, line);
            istringstream iss3(line);
            iss3 >> lr >> momentum;
            optimizer = new SGD(lr, momentum);
            bias_optimizer = new SGD(lr, momentum);
        }
        else if(optimizer_type == "Adam") {
            double lr, mu, rho;
            getline(file, line);
            istringstream iss3(line);
            iss3 >> lr >> mu >> rho;
            optimizer = new Adam(lr, mu, rho);
            bias_optimizer = new Adam(lr, mu, rho);
        }
        else {
            optimizer = nullptr;
            bias_optimizer = nullptr;
        }
        weights = Tensor(1, output_size, input_size);
        bias = Tensor(1, output_size, 1);
        getline(file, line);
        istringstream iss3(line);
        for(int i = 0; i < output_size; i++) {
            for(int j = 0; j < input_size; j++) {
                iss3 >> weights(0, i, j);
            }
        }
        getline(file, line);
        istringstream iss4(line);
        for(int i = 0; i < output_size; i++) {
            iss4 >> bias(0, i, 0);
        }
    }
    file.close();
}