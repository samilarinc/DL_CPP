#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "Dense.hpp"
#include "Tensor.hpp"
#include "SGD.hpp"
#include "BaseOptimizer.hpp"
#include "L1Loss.hpp"
#include "L2Loss.hpp"
#include "CrossEntropyLoss.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Scheduler.hpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(pyflow, m){
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, double>())
        .def(py::init<int, int, int, double, double>())
        .def(py::init<const Tensor&>())
        .def(py::init<vector<vector<vector<double>>>>())
        .def(py::init<vector<vector<double>>>())
        .def(py::init<vector<double>>())
        .def("__len__", &Tensor::size)
        .def("__add__", py::overload_cast<const Tensor&>(&Tensor::operator+))
        .def("__add__", py::overload_cast<double>(&Tensor::operator+))
        .def("__sub__", py::overload_cast<const Tensor&>(&Tensor::operator-))
        .def("__sub__", py::overload_cast<double>(&Tensor::operator-))
        .def("__mul__", py::overload_cast<const Tensor&>(&Tensor::operator*))
        .def("__mul__", py::overload_cast<double>(&Tensor::operator*))
        .def("__truediv__", py::overload_cast<const Tensor&>(&Tensor::operator/))
        .def("__truediv__", py::overload_cast<double>(&Tensor::operator/))
        .def("__pow__", py::overload_cast<double>(&Tensor::power))
        .def("__iadd__", py::overload_cast<const Tensor&>(&Tensor::operator+=))
        .def("__iadd__", py::overload_cast<double>(&Tensor::operator+=))
        .def("__isub__", py::overload_cast<const Tensor&>(&Tensor::operator-=))
        .def("__isub__", py::overload_cast<double>(&Tensor::operator-=))
        .def("__imul__", py::overload_cast<const Tensor&>(&Tensor::operator*=))
        .def("__imul__", py::overload_cast<double>(&Tensor::operator*=))
        .def("__itruediv__", py::overload_cast<const Tensor&>(&Tensor::operator/=))
        .def("__itruediv__", py::overload_cast<double>(&Tensor::operator/=))
        .def("__pow__", py::overload_cast<double>(&Tensor::power))
        .def("power", py::overload_cast<double>(&Tensor::power))
        .def("abs", &Tensor::abs)
        .def("sign", &Tensor::sign)
        .def("setitem", py::overload_cast<int, int, int, double>(&Tensor::setitem))
        .def("setitem", py::overload_cast<int, int, const Tensor&>(&Tensor::setitem))
        .def("setitem", py::overload_cast<int, const Tensor&>(&Tensor::setitem))
        .def("rows", &Tensor::getRows)
        .def("cols", &Tensor::getCols)
        .def("batch_size", &Tensor::getBatchsize)
        .def("__repr__", &Tensor::toString)
        .def("transpose", &Tensor::transpose)
        .def("dot_product", &Tensor::dot_product)
        .def_readonly("shape", &Tensor::shape)
        .def("copy", &Tensor::copy)
        .def("sum", &Tensor::sum)
        .def("tolist", &Tensor::tolist)
        .def("getitem", py::overload_cast<int>(&Tensor::getitem, py::const_))
        .def("getitem", py::overload_cast<int, int>(&Tensor::getitem, py::const_))
        .def("getitem", py::overload_cast<int, int, int>(&Tensor::getitem, py::const_))
        .def("__call__", [](Tensor& t, int batch_size, int row, int col) {
            return t(batch_size, row, col);
        })
        .def("__call__", [](Tensor& t, int batch_size, int row, int col) {
            return t(batch_size, row, col);
        })
        .def("__repr__", [](Tensor& t) {
            return t.toString();
        })
        ;
    
    py::class_<Dense>(m, "FullyConnected")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def(py::init<int, int, double, string>())
        .def(py::init<int, int, double, string, double>())
        .def(py::init<int, int, double, string, double, double>())
        .def(py::init<int, int, double, string, double, double, double>())
        .def("forward", &Dense::forward)
        .def("backward", &Dense::backward)
        .def("save", &Dense::save)
        .def("load", &Dense::load)
        .def_readwrite("weights", &Dense::weights)
        .def_readwrite("bias", &Dense::bias)
        .def_readwrite("trainable", &Dense::trainable)
        .def_readwrite("gradient_weights", &Dense::gradient_weights)
        .def_readwrite("gradient_bias", &Dense::gradient_bias)
        ;

    py::class_<SGD>(m, "SGD")
        .def(py::init<double>())
        .def("calculate_update", &SGD::calculate_update)
        .def("learning_rate", &SGD::getLearningRate)
        .def("momentum", &SGD::getMomentum)
        .def("addScheduler", &SGD::addScheduler)
        ;
    py::class_<Adam>(m, "Adam")
        .def(py::init<double, double, double>())
        .def("calculate_update", &Adam::calculate_update)
        .def("learning_rate", &Adam::getLearningRate)
        .def("beta1", &Adam::getMu)
        .def("beta2", &Adam::getRho)
        .def("addScheduler", &Adam::addScheduler)
        ;
    py::class_<L1Loss>(m, "L1Loss")
        .def(py::init<>())
        .def("forward", &L1Loss::forward)
        .def("backward", &L1Loss::backward)
        ;
    py::class_<L2Loss>(m, "L2Loss")
        .def(py::init<>())
        .def("forward", &L2Loss::forward)
        .def("backward", &L2Loss::backward)
        ;
    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward)
        ;
    py::class_<ReLU>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward)
        .def("save", &ReLU::save)
        .def("load", &ReLU::load)
        ;
    py::class_<Sigmoid>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward)
        .def("backward", &Sigmoid::backward)
        .def("save", &Sigmoid::save)
        .def("load", &Sigmoid::load)
        ;
    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init<int, double>())
        .def(py::init<vector<int>, double>())
        .def(py::init<double>())
        .def(py::init<double, int>())
        .def(py::init<double, double, int>())
        .def("step", &Scheduler::step)
        ;
    py::enum_<SchedulerType>(m, "SchedulerType")
        .value("STEP_LR", SchedulerType::STEP_LR)
        .value("MULTI_STEP_LR", SchedulerType::MULTI_STEP_LR)
        .value("EXPONENTIAL_LR", SchedulerType::EXPONENTIAL_LR)
        .value("POLYNOMIAL_LR", SchedulerType::POLYNOMIAL_LR)
        .value("LINEAR_LR", SchedulerType::LINEAR_LR)
        .export_values()
        ;
}