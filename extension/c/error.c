#include <math.h>
#include <stdio.h>

#include "../error.h"
#include "../matrix.h"

double meanSquaredError(double *expectedYs, double *actualYs, int n) {
    double sum = 0;

    for (int i = 0; i < n; i++) {
        double difference = expectedYs[i] - actualYs[i];
        sum += difference * difference;
    }

    return sum / (double)n;
}

double crossEntropyLoss(double *expectedYs, double *actualYs, int n) {
    long double sum = 0;

    for (int i = 0; i < n; i++) {
        sum += expectedYs[i] * log2(actualYs[i]);
    }

    return -(double)sum;
}

matrix2d_t *dMeanSquaredError(matrix2d_t *expectedY, matrix2d_t *actualY) {
    return matrixSubtract(expectedY, actualY);
}

matrix2d_t *dCrossEntropyLoss(matrix2d_t *expectedY, matrix2d_t *actualY) {
    return NULL;
}