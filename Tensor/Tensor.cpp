#include "Tensor.hpp"

Tensor::Tensor() {
    batch_size = 0;
    rows = 0;
    cols = 0;
    data = nullptr;
    shape = make_tuple(0, 0, 0);
}

Tensor::Tensor(int batch_size, int rows, int cols) {
    this->batch_size = batch_size;
    this->rows = rows;
    this->cols = cols;
    this->data = new double[batch_size * rows * cols]();
    this->shape = make_tuple(batch_size, rows, cols);
}

Tensor::Tensor(int batch_size, int rows, int cols, double constant){
    this->batch_size = batch_size;
    this->rows = rows;
    this->cols = cols;
    this->shape = make_tuple(batch_size, rows, cols);
    data = new double[batch_size * rows * cols]();
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] = constant;
    }
}

Tensor::Tensor(int batch_size, int rows, int cols, double min, double max, string initializer) { 
    this->batch_size = batch_size;
    this->rows = rows;
    this->cols = cols;
    this->shape = make_tuple(batch_size, rows, cols);
    data = new double[batch_size * rows * cols]();
    
    mt19937 rng(0);
    uniform_real_distribution<double> dist(min, max);

    if (strcmp(initializer.c_str(), "uniform") == 0){
        for (int i = 0; i < batch_size * rows * cols; i++)
            data[i] = dist(rng);
    }
    else if (strcmp(initializer.c_str(), "xavier") == 0) {
        double scale = sqrt(6.0 / (rows + cols));
        for (int i = 0; i < batch_size * rows * cols; i++)
            data[i] = dist(rng) * scale;
    }
    else if (strcmp(initializer.c_str(), "he") == 0) {
        double scale = sqrt(2.0 / (rows));
        for (int i = 0; i < batch_size * rows * cols; i++)
            data[i] = dist(rng) * scale;
    }
    else if (strcmp(initializer.c_str(), "constant") == 0) {
        for (int i = 0; i < batch_size * rows * cols; i++)
            data[i] = min;
    }
    else {
        printf("Invalid initializer\n");
    }
}

Tensor::Tensor(vector<vector<vector<double>>> data) {
    batch_size = data.size();
    rows = data[0].size();
    cols = data[0][0].size();
    this->shape = make_tuple(batch_size, rows, cols);
    this->data = new double[batch_size * rows * cols]();
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < rows; j++) {
            for(int k = 0; k < cols; k++) {
                this->data[i * rows * cols + j * cols + k] = data[i][j][k];
            }
        }
    }
}

Tensor::Tensor(vector<vector<double>> data) {
    batch_size = 1;
    rows = data.size();
    cols = data[0].size();
    this->shape = make_tuple(batch_size, rows, cols);
    this->data = new double[batch_size * rows * cols]();
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            this->data[i * cols + j] = data[i][j];
        }
    }
}

Tensor::Tensor(vector<double> data) {
    batch_size = 1;
    rows = data.size();
    cols = 1;
    this->shape = make_tuple(batch_size, rows, cols);
    this->data = new double[batch_size * rows * cols]();
    for(int i = 0; i < rows; i++) {
        this->data[i] = data[i];
    }
}

Tensor::Tensor(const Tensor& other) {
    batch_size = other.batch_size;
    rows = other.rows;
    cols = other.cols;
    this->shape = make_tuple(batch_size, rows, cols);
    data = new double[batch_size * rows * cols];
    for (int i = 0; i < batch_size * rows * cols; i++) {
        data[i] = other.data[i];
    }
}

Tensor::~Tensor() {
    delete[] data;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data;
        batch_size = other.batch_size;
        rows = other.rows;
        cols = other.cols;
        data = new double[batch_size * rows * cols];
        for (int i = 0; i < batch_size * rows * cols; i++) {
            data[i] = other.data[i];
        }
    }
    return *this;
}

double& Tensor::operator()(int batch, int row, int col) {
    return data[batch * rows * cols + row * cols + col];
}

double Tensor::operator()(int batch, int row, int col) const {
    return data[batch * rows * cols + row * cols + col];
}

Tensor Tensor::operator+(const Tensor& other) {
    if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
        throw invalid_argument("Dimensions of the two tensors must be equal.");
    }
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) {
    if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
        throw invalid_argument("Dimensions of the two tensors must be equal.");
    }
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) {
    if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
        throw invalid_argument("Dimensions of the two tensors must be equal.");
    }
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) {
    if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
        throw invalid_argument("Dimensions of the two tensors must be equal.");
    }
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        if(other.data[i] == 0){
            result.data[i] = data[i] / (other.data[i] + 1e-9);
        }
        else{
            result.data[i] = data[i] / other.data[i];
        }
    }
    return result;
}

Tensor Tensor::operator+(double constant) {
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = data[i] + constant;
    }
    return result;
}

Tensor Tensor::operator-(double constant) {
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = data[i] - constant;
    }
    return result;
}

Tensor Tensor::operator*(double constant) {
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = data[i] * constant;
    }
    return result;
}

Tensor Tensor::operator/(double constant) {
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = data[i] / constant;
    }
    return result;
}

Tensor Tensor::power(double exponent) {
    Tensor result(batch_size, rows, cols);
    for (int i = 0; i < batch_size * rows * cols; i++) {
        result.data[i] = pow(data[i], exponent);
    }
    return result;
}

int Tensor::getRows() const {
    return rows;
}

int Tensor::getCols() const {
    return cols;
}

int Tensor::getBatchsize() const {
    return batch_size;
}

string Tensor::toString() const {
    string output = "";
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                output += to_string(data[i * rows * cols + j * cols + k]) + " ";
            }
            if(j != rows - 1) {
                output += "\n";
            }
        }
        if(i != batch_size - 1) {
            output += "\n\n";
        }
    }
    return output;
}

Tensor Tensor::transpose(){
    Tensor output(batch_size, cols, rows);
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < rows; j++){
            for(int k = 0; k < cols; k++){
                output(i, k, j) = (*this)(i, j, k);
            }
        }
    }
    return output;
}

Tensor Tensor::dot_product(const Tensor& other){
    if(cols != other.rows){
        printf("Error: dot_product: cols != other.rows");
        return Tensor();
    }
    Tensor output(batch_size, rows, other.cols);
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < rows; j++){
            for(int k = 0; k < other.cols; k++){
                for(int l = 0; l < cols; l++){
                    output(i, j, k) += (*this)(i, j, l) * other(i, l, k);
                }
            }
        }
    }
    return output;
}

double Tensor::sum() const{
    double output = 0;
    for(int i = 0; i < batch_size * rows * cols; i++){
        output += data[i];
    }
    return output;
}

Tensor Tensor::sign() const{
    Tensor output(batch_size, rows, cols);
    for(int i = 0; i < batch_size * rows * cols; i++){
        if(data[i] > 0){
            output.data[i] = 1;
        } else if(data[i] < 0){
            output.data[i] = -1;
        } else {
            output.data[i] = 0;
        }
    }
    return output;
}

Tensor Tensor::abs() const{
    Tensor output(batch_size, rows, cols);
    for(int i = 0; i < batch_size * rows * cols; i++){
        output.data[i] = data[i] > 0 ? data[i] : -data[i];
    }
    return output;
}

Tensor Tensor::copy() const{
    Tensor output(batch_size, rows, cols);
    std::copy(data, data + batch_size * rows * cols, output.data);
    return output;
}

Tensor Tensor::getitem(int num) const{
    if(batch_size == 1) {
        Tensor output(1, cols, 1);
        for(int i = 0; i < cols; i++){
            output(0, i, 0) = (*this)(0, num, i);
        }
        return output;
    }
    Tensor output(1, rows, cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            output(0, i, j) = (*this)(num, i, j);
        }
    }
    return output;
}

Tensor Tensor::getitem(int num1, int num2) const{
    Tensor output(batch_size, 1, 1);
    for(int i = 0; i < batch_size; i++){
        output(i, 0, 0) = (*this)(i, num1, num2);
    }
    return output;
}

double Tensor::getitem(int num1, int num2, int num3) const{
    return (*this)(num1, num2, num3);
}

void Tensor::setitem(int num, const Tensor& other){
    if(batch_size == 1) {
        for(int i = 0; i < cols; i++){
            (*this)(0, num, i) = other(0, i, 0);
        }
        return;
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            (*this)(num, i, j) = other(0, i, j);
        }
    }
}

void Tensor::setitem(int num1, int num2, const Tensor& other){
    for(int i = 0; i < batch_size; i++){
        (*this)(i, num1, num2) = other(i, 0, 0);
    }
}

void Tensor::setitem(int num1, int num2, double value){
    for(int i = 0; i < batch_size; i++){
        (*this)(i, num1, num2) = value;
    }
}

void Tensor::setitem(int num1, int num2, int num3, double value){
    (*this)(num1, num2, num3) = value;
}

vector<vector<vector<double>>> Tensor::tolist() const{
    vector<vector<vector<double>>> output(batch_size, vector<vector<double>>(rows, vector<double>(cols)));
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < rows; j++){
            for(int k = 0; k < cols; k++){
                output[i][j][k] = (*this)(i, j, k);
            }
        }
    }
    return output;
}

Tensor& Tensor::operator+=(const Tensor& tensor2){
    if(batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols){
        printf("Error: operator+=: batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols");
        return *this;
    }
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] += tensor2.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& tensor2){
    if(batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols){
        printf("Error: operator-=: batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols");
        return *this;
    }
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] -= tensor2.data[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& tensor2){
    if(batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols){
        printf("Error: operator*=: batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols");
        return *this;
    }
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] *= tensor2.data[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& tensor2){
    if(batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols){
        printf("Error: operator/=: batch_size != tensor2.batch_size || rows != tensor2.rows || cols != tensor2.cols");
        return *this;
    }
    for(int i = 0; i < batch_size * rows * cols; i++){
        if(tensor2.data[i] == 0){
            data[i] /= tensor2.data[i] + 1e-10;            
        }
        else{
            data[i] /= tensor2.data[i];
        }
    }
    return *this;
}

Tensor& Tensor::operator+=(double value){
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] += value;
    }
    return *this;
}

Tensor& Tensor::operator-=(double value){
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] -= value;
    }
    return *this;
}

Tensor& Tensor::operator*=(double value){
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] *= value;
    }
    return *this;
}

Tensor& Tensor::operator/=(double value){
    if(value == 0){
        value += 1e-10;
    }
    for(int i = 0; i < batch_size * rows * cols; i++){
        data[i] /= value;
    }
    return *this;
}

int Tensor::size() const{
    return batch_size * rows * cols;
}