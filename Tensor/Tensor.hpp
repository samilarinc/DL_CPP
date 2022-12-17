#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <math.h>
#include <string.h>

#include <tuple>
#include <vector>
#include <string>
#include <exception>
#include <random>
#include <cmath>

using namespace std;

class Tensor {
public:
    Tensor();
    Tensor(int, int, int);
    Tensor(int, int, int, double);
    Tensor(int, int, int, double, double, string); 
    Tensor(const Tensor&);
    Tensor(vector<vector<vector<double>>>);
    Tensor(vector<vector<double>>);
    Tensor(vector<double>);
    ~Tensor();
    Tensor transpose();
    Tensor dot_product(const Tensor&);
    Tensor& operator=(const Tensor&);
    double& operator()(int, int, int);
    double operator()(int, int, int) const;
    Tensor operator+(const Tensor&);
    Tensor operator-(const Tensor&);
    Tensor operator*(const Tensor&);
    Tensor operator/(const Tensor&);
    Tensor& operator+=(const Tensor&);
    Tensor& operator-=(const Tensor&);
    Tensor& operator*=(const Tensor&);
    Tensor& operator/=(const Tensor&);
    Tensor operator+(double);
    Tensor operator-(double);
    Tensor operator*(double);
    Tensor operator/(double);
    Tensor& operator+=(double);
    Tensor& operator-=(double);
    Tensor& operator*=(double);
    Tensor& operator/=(double);
    Tensor power(double);
    int getRows() const;
    int getCols() const;
    int getBatchsize() const;
    string toString() const;
    double sum() const;
    Tensor sign() const;
    Tensor abs() const;
    Tensor copy() const;
    Tensor getitem(int) const;
    Tensor getitem(int, int) const;
    double getitem(int, int, int) const;
    void setitem(int, int, int, double);
    void setitem(int, int, double);
    void setitem(int, int, const Tensor&);
    void setitem(int, const Tensor&);
    int size() const;
    vector<vector<vector<double>>> tolist() const;
    tuple<int, int, int> shape;
// private:
    int rows;
    int cols;
    int batch_size;
    double* data;
};

#endif // TENSOR_HPP