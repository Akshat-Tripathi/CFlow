

#include "../activation.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../matrix.h"
enum activationFunction getDeriv(enum activationFunction func) {
    switch (func) {

        case LRELU:   return LRELU_PRIME;
        case LINEAR:  return LINEAR_PRIME;
        case RELU: 	  return RELU_PRIME;
        case TANH:    return TANH_PRIME;
        case SIGMOID: return SIGMOID_PRIME;
        default:      return -1;
    }
}
#define ALPHA 0.2

double relu(double x) {
    return (x > 0) ? x : 0;
}

double reluPrime(double x) {
    assert(x != 0);
    return (x > 0);
}

//"lRelu" is actually parametric ReLu
//Real lRelu is max(x, 0.01*x), but pReLu is still valid for our purposes
double lRelu(double x) {
    return ((ALPHA * x) > x) ? ALPHA * x : x;
}

double lReluPrime() {
    return (ALPHA > 1) ? ALPHA : 1;
}

double linear(double x) {
    return x;
}

double linearPrime() {
    return 1;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

//PRE: x is already sigmoided
double sigmoidPrime(double x) {
    return x * (1 - x);
    //Return statements are equivalent, but second return statement exceeds double size limit
    //return exp(x) / ((exp(x) + 1) * (exp(x) + 1));
}

double tanhActive(double x) {
    return tanh(x);
}

double tanhPrime(double x) {
    return 1 - (x * x);
}

static double softmaxSum(matrix3d_t *matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            sum += exp(matrixGet(matrix, i, j));
        }
    }
}

matrix2d_t *softmax(matrix2d_t *matrix) {
    double sum = softmaxSum(matrix);
    matrix2d_t *result = matrixCreate(matrix->nRows, matrix->nCols);
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            matrixSet(result, i, j, exp(matrixGet(matrix, i, j)) / sum);
        }
    }
    return result;
}

matrix2d_t *softmaxPrime(matrix2d_t *matrix) {    
    double sum = softmaxSum(matrix);
    matrix2d_t *result = matrixCreate(matrix->nRows, matrix->nCols);
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            if (i == j) {
                matrixSet(result, i, j, (exp(matrixGet(matrix, i, j)) / sum) * (1 - (exp(matrixGet(matrix, j, i)) / sum)));
            } else {
                matrixSet(result, i, j, (exp(matrixGet(matrix, i, j)) / sum) * (- (exp(matrixGet(matrix, j, i)) / sum)));
            }
        }
    }
    return result;
}
