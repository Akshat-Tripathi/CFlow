#include <math.h>
#include <stdio.h>

#include "../error.h"
#include "../matrix.h"

double meanSquaredError(matrix2d_t *expectedYs, matrix2d_t *actualYs) {
    matrix2d_t *difference = matrixSubtract(expectedYs, actualYs);
    matrix2d_t *diffSqaured = matrixMultiplyElementWise(difference, difference);
    
    double sum = 0;
    for (int i = 0; i < actualYs->nRows; i++) {
        for (int j = 0; j < actualYs->nCols; j++) {
            sum += matrixGet(diffSqaured, i, j);
        }
    }
    return sum / (double) (actualYs->nCols + actualYs->nRows);
}


double crossEntropyLoss(matrix2d_t *expectedYs, matrix2d_t *actualYs) {
    matrix2d_t *actualLog = matrixElementWise(actualYs, log2); 
    matrix2d_t *loss = matrixMultiplyElementWise(expectedYs, actualLog);
    long double sum = 0;

    for (int i = 0; i < actualYs->nRows; i++) {
        for (int j = 0; j < actualYs->nCols; j++) {
            sum += matrixGet(loss, i, j);
        }
    }

    return -(double)sum;
}


matrix2d_t *dMeanSquaredError(matrix2d_t *expectedY, matrix2d_t *actualY) {
    return matrixSubtract(expectedY, actualY);
}

matrix2d_t *dCrossEntropyLoss(matrix2d_t *expectedY, matrix2d_t *actualY) {
    return NULL;
}