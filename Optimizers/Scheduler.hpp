#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include <vector>
#include <math.h>
#include <stdio.h>

enum class SchedulerType
{
    NOT_SET,            // NO SCHEDULER
    STEP_LR,            // Multiply by constant in each x step 
                        // x, constant
    MULTI_STEP_LR,      // Multiply by constant in x, y... steps
                        // list of epochs, constant
    EXPONENTIAL_LR,     // Multiply by constant in each epoch
                        // constant
    POLYNOMIAL_LR,      // Multiply by (1.0 - last_epoch/total_iters)^power
                        // power
    LINEAR_LR,          // start_constant * start_lr is the initial learning rate
                        // end_constant * start_lr is the final learning rate (after total_iters)
                        // learning rate is linearly interpolated between the two
                        // start_constant, end_constant, total_iters
};

class Scheduler
{
public:
    Scheduler(); // Default constructor
    Scheduler(int, double); // STEP_LR
    Scheduler(std::vector<int>, double); // MULTI_STEP_LR
    Scheduler(double); // EXPONENTIAL_LR
    Scheduler(double, int); // POLYNOMIAL_LR
    Scheduler(double, double, int); // LINEAR_LR
    ~Scheduler();
    void step(int, double&);
    SchedulerType getType() const;

private:
    double constant;
    int frequency;
    int* list_of_epochs = nullptr;
    double end_constant;
    int total_iters;
    SchedulerType type;
};

#endif // SCHEDULER_HPP