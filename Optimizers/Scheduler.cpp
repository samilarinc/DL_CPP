#include "Scheduler.hpp"

Scheduler::Scheduler(){
    this->type = SchedulerType::NOT_SET;
}

Scheduler::Scheduler(int frequency, double constant) {
    this->frequency = frequency;
    this->constant = constant;
    this->type = SchedulerType::STEP_LR;
}

Scheduler::Scheduler(std::vector<int> list_of_epochs, double constant) {
    if(list_of_epochs.size() > 0) {
        this->list_of_epochs = new int[list_of_epochs.size() + 1]; // 1 for length of the list
        this->list_of_epochs[0] = list_of_epochs.size();
        for(int i = 0; i < list_of_epochs.size(); i++) {
            this->list_of_epochs[i+1] = list_of_epochs[i];
        }
    }
    else{
        printf("Error: list_of_epochs is empty");
        this->list_of_epochs = new int[1];
        this->list_of_epochs[0] = 0;
    }
    this->constant = constant;
    this->type = SchedulerType::MULTI_STEP_LR;
}

Scheduler::Scheduler(double constant) {
    this->constant = constant;
    this->type = SchedulerType::EXPONENTIAL_LR;
}

Scheduler::Scheduler(double constant, int total_iters) {
    this->constant = constant;
    this->total_iters = total_iters;
    this->type = SchedulerType::POLYNOMIAL_LR;
}

Scheduler::Scheduler(double start_constant, double end_constant, int total_iters) {
    this->constant = start_constant;
    this->end_constant = end_constant;
    this->total_iters = total_iters;
    this->type = SchedulerType::LINEAR_LR;
}

Scheduler::~Scheduler() {
    if(this->list_of_epochs != nullptr) {
        delete[] this->list_of_epochs;
    }
}

void Scheduler::step(int epoch, double& learning_rate) {
    switch(this->type) {
        case SchedulerType::STEP_LR:
            if(epoch % this->frequency == 0) {
                learning_rate *= this->constant;
            }
            break;
        case SchedulerType::MULTI_STEP_LR:
            for(int i = 1; i <= this->list_of_epochs[0]; i++) {
                if(epoch == this->list_of_epochs[i]) {
                    learning_rate *= this->constant;
                }
            }
            break;
        case SchedulerType::EXPONENTIAL_LR:
            learning_rate *= this->constant;
            break;
        case SchedulerType::POLYNOMIAL_LR:
            learning_rate *= pow(1.0 - (double)epoch / (double)this->total_iters, this->constant);
            break;
        case SchedulerType::LINEAR_LR:
            learning_rate = learning_rate * (this->constant + (this->end_constant - this->constant) * (double)epoch / (double)this->total_iters);
            break;
    }
}

SchedulerType Scheduler::getType() const {
    return this->type;
}