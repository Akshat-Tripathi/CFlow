#ifndef _matrix_h_
#define _matrix_h_

#include <stdbool.h>

enum matrixFunction {
    ADD,
    ACTIVATION,
    SUBTRACT,
    MULTIPLY,
    DOT,
    CONVOLUTION,
    DECONVOLUTION,
    DILATE,
    ROTATE,
    MAX_POOLING,
    AVERAGE_POOLING,
    TRANSPOSE,  //PRE: Won't ever be in a forward pass
    FLATTEN
};

typedef struct matrix2d {
    double **data;
    int nRows;
    int nCols;
} matrix2d_t;

typedef struct matrix3d {
    double ***data;
    int nRows;
    int nCols;
    int nDepth;
} matrix3d_t;


#include "activation.h"
matrix2d_t *matrixCreate(int nRows, int nCols);
matrix3d_t *matrix3DCreate(int nRows, int nCols, int nDepth);
double matrixGet(matrix2d_t *matrix, int row, int col);
void matrixSet(matrix2d_t *matrix, int row, int col, double value);
double randFloat();
void matrixRandomise(matrix2d_t *matrix);

matrix2d_t *matrixOperation(matrix2d_t *matrix1, matrix2d_t *matrix2, double (*func)(double, double));
matrix2d_t *matrixAdd(matrix2d_t *matrix1, matrix2d_t *matrix2);
matrix2d_t *matrixSubtract(matrix2d_t *matrix1, matrix2d_t *matrix2);
matrix2d_t *matrixMultiplyElementWise(matrix2d_t *matrix1, matrix2d_t *matrix2);
matrix2d_t *matrixRotate(matrix2d_t *matrix, double rotationDegrees, int planeRow, int planeCol);
matrix2d_t *matrixDilate(matrix2d_t *matrix, int dilation);
matrix2d_t *matrixSquareRoot(matrix2d_t *matrix);
matrix2d_t *matrixElementWise(matrix2d_t *matrix, double (*func)(double));
matrix2d_t *matrixDivisionElementWise(matrix2d_t *matrix1, matrix2d_t *matrix2);

matrix2d_t *matrixScalarProduct(matrix2d_t *matrix, double scalar);
matrix2d_t *matrixDotProduct(matrix2d_t *matrix1, matrix2d_t *matrix2);
matrix2d_t *matrixTranspose(matrix2d_t *matrix);

matrix2d_t *matrixActiveFunc(matrix2d_t *matrix, enum activationFunction func);
matrix2d_t *matrixConvolution(matrix2d_t *matrix, matrix2d_t *kernel, int stride, int padding);
matrix2d_t *matrixDeconvolution(matrix2d_t *matrix, matrix2d_t *kernel, int stride, int padding);
matrix3d_t *matrix3DConvolution(matrix3d_t *inputs, matrix3d_t *kernels, int stride, int padding);

matrix2d_t *matrixMaxPooling(matrix2d_t *matrix, matrix2d_t *gradient, int stride, int filterSize);
matrix2d_t *matrixAveragePooling(matrix2d_t *matrix, matrix2d_t *gradient, int stride, int filterSize);

bool areMatrixesEqual(matrix2d_t *matrix1, matrix2d_t *matrix2, double tolerance);

matrix2d_t *matrixFlatten(matrix3d_t *matrix);
double* flatten2d(matrix2d_t *matrix);
matrix3d_t *matrixUnflatten(matrix2d_t *matrix, int nRows, int nCols, int nDepth);

void matrixPrint(matrix2d_t *matrix);

void matrixFree(matrix2d_t *matrix);

#endif
