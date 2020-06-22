#ifndef _activation_h_ 
#define _activation_h_

enum activationFunction {
    RELU, 
    RELU_PRIME,
    LRELU,
    LRELU_PRIME,
    LINEAR,
    LINEAR_PRIME,
    SIGMOID, 
    SIGMOID_PRIME,
    TANH,
    TANH_PRIME,
    SOFTMAX,
    SOFTMAX_PRIME
};
#include "matrix.h"
enum activationFunction getDeriv(enum activationFunction func);
double relu(double x);
double reluPrime(double x);
double lRelu(double x);
double lReluPrime();
double linear(double x);
double linearPrime();
double sigmoid(double x);
double sigmoidPrime(double x);
double tanhActive(double x);
double tanhPrime(double x);
matrix2d_t *softmax(matrix2d_t *matrix);
matrix2d_t *softmaxPrime(matrix2d_t *matrix);    

#endif