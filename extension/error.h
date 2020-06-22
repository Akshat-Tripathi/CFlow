#ifndef _error_h_
#define _error_h_ 

#include "matrix.h"

enum errorFunction {
    MSE,
    CSL
};

double meanSquaredError(double *expectedYs, double *actualYs, int n);
double crossEntropyLoss(double *expectedYs, double *actualYs, int n);
matrix2d_t *dMeanSquaredError(matrix2d_t *expectedY, matrix2d_t *actualY);
matrix2d_t *dCrossEntropyLoss(matrix2d_t *expectedY, matrix2d_t *actualY);

#endif
