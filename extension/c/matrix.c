#include "../matrix.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "../activation.h"

matrix2d_t *matrixCreate(int nRows, int nCols) {
    matrix2d_t *matrix = malloc(sizeof(matrix2d_t));

    matrix->nRows = nRows;
    matrix->nCols = nCols;

    double **data = calloc(nRows, sizeof(double*));
    for (int i = 0; i < nRows; i++) {
        data[i] = calloc(nCols, sizeof(double));
    }
    matrix->data = data;

    return matrix;
}

matrix3d_t *matrix3DCreate(int nRows, int nCols, int nDepth) {
    matrix3d_t *matrix = malloc(sizeof(matrix3d_t));

    matrix->nRows = nRows;
    matrix->nCols = nCols;
    matrix->nDepth = nDepth;

    double ***data = calloc(nRows, sizeof(double*));
    for (int i = 0; i < nRows; i++) {
        data[i] = calloc(nCols, sizeof(double*));
        for (int j = 0; j < nCols; j++) {
            data[i][j] = calloc(nDepth, sizeof(double));
        }
    }
    matrix->data = data;
    return matrix;
}

double matrixGet(matrix2d_t *matrix, int row, int col) {
    assert(matrix);
    return matrix->data[row][col];
}

void matrixSet(matrix2d_t *matrix, int row, int col, double value) {
    assert(matrix);
    matrix->data[row][col] = value;
}

matrix2d_t *matrixOperation(matrix2d_t *matrix1, matrix2d_t *matrix2, double (*func)(double, double)) {
    assert(matrix1);
    assert(matrix2);
    if (matrix1->nCols != matrix2->nCols || matrix1->nRows != matrix2->nRows) {
        return NULL;
    }

    int nCols = matrix1->nCols;
    int nRows = matrix1->nRows;
    matrix2d_t *new_matrix = matrixCreate(nRows, nCols);

    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrixSet(new_matrix, i, j, func(matrixGet(matrix1, i, j), matrixGet(matrix2, i, j)));
        }
    }
    return new_matrix;
}

static double add(double a, double b) {
    return a + b;
}

matrix2d_t *matrixAdd(matrix2d_t *matrix1, matrix2d_t *matrix2) {
    return matrixOperation(matrix1, matrix2, add);
}

static double subtract(double a, double b) {
    return a - b;
}

matrix2d_t *matrixSubtract(matrix2d_t *matrix1, matrix2d_t *matrix2) {
    return matrixOperation(matrix1, matrix2, subtract);
}

static double multiply(double a, double b) {
    return a * b;
}

static double divison(double a, double b) {
    return a / b;
}

matrix2d_t *matrixMultiplyElementWise(matrix2d_t *matrix1, matrix2d_t *matrix2) {
    return matrixOperation(matrix1, matrix2, multiply);
}

matrix2d_t *matrixDivisionElementWise(matrix2d_t *matrix1, matrix2d_t *matrix2) {
    return matrixOperation(matrix1, matrix2, divison);
}

matrix2d_t *matrixScalarProduct(matrix2d_t *matrix, double scalar) {
    assert(matrix);
    int nCols = matrix->nCols;
    int nRows = matrix->nRows;
    matrix2d_t *new_matrix = matrixCreate(nRows, nCols);
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrixSet(new_matrix, i, j, scalar * matrixGet(matrix, i, j));
        }
    }
    return new_matrix;
}

matrix2d_t *matrixDotProduct(matrix2d_t *matrix1, matrix2d_t *matrix2) {
    assert(matrix1);
    assert(matrix2);

    if (matrix1->nCols != matrix2->nRows) {
        if (matrix2->nCols == matrix1->nRows) {
            return matrixDotProduct(matrix2, matrix1);
        } else {
            perror("Matrices have incompatible dimensions\n");
            exit(EXIT_FAILURE);
        }
    }
    
    int nRows = matrix1->nRows;
    int nCols = matrix2->nCols;
    int nShared = matrix1->nCols;

    matrix2d_t *output = matrixCreate(nRows, nCols);
    for (int i = 0; i < nRows; i++) {
        for (int k = 0; k < nShared; k++) {
            double matrix1Value = matrixGet(matrix1, i, k);
            if (matrix1Value != 0) {
                for (int j = 0; j < nCols; j++) {
                    matrixSet(output, i, j, matrixGet(output, i, j) + matrix1Value * matrixGet(matrix2, k, j));
                }
            }
        }
    }
    return output;
}

matrix2d_t *matrixTranspose(matrix2d_t *matrix) {
    int nCols = matrix->nCols;
    int nRows = matrix->nRows;
    matrix2d_t *result = matrixCreate(nCols, nRows);
    for (int i = 0; i < nCols; i++) {
        for (int j = 0; j < nRows; j++) {
            matrixSet(result, i, j, matrixGet(matrix, j, i));
        }
    }
    return result;
}

// PRE: PlaneRow and planeCol are the row/column in the rotation matrix
//      (zero indexed) where cos and sin will be, representing the plane of rotation
// POST: Input matrix rotated by provided angle
matrix2d_t *matrixRotate(matrix2d_t *matrix, double rotationDegrees, int planeRow, int planeCol) {
    assert(planeRow != planeCol); //cannot rotate in the same axis
    double radians = (rotationDegrees / 180) * acos(-1.0);
    matrix2d_t *rotation = matrixCreate(matrix->nRows, matrix->nRows);
    for (int i = 0; i < matrix->nRows; i++) {
        matrixSet(rotation, i, i, 1);
    }
    matrixSet(rotation, planeRow, planeRow, cos(radians));
    matrixSet(rotation, planeCol, planeCol, cos(radians));
    matrixSet(rotation, planeRow, planeCol, -sin(radians));
    matrixSet(rotation, planeCol, planeRow, sin(radians));
    return matrixDotProduct(rotation, matrix);
}

//PRE: matrix is initialised and alpha is only used for lRelu
//POST: applies provided activation function element wise on matrix
matrix2d_t *matrixActiveFunc(matrix2d_t *matrix, enum activationFunction func) {
    int nCols = matrix->nCols;
    int nRows = matrix->nRows;
    matrix2d_t *new_matrix = matrixCreate(nRows, nCols);
    bool twoArgFunc = true;
    double (*activeFunc)(double);
    double (*activeFuncSing)();
    switch (func) {
        case RELU:
            activeFunc = &relu;
            break;
        case RELU_PRIME:
            activeFunc = &reluPrime;
            break;
        case LRELU:
            activeFunc = &lRelu;
            break;
        case LRELU_PRIME:
            activeFuncSing = &lReluPrime;
            twoArgFunc = false;
            break;
        case LINEAR:
            activeFunc = &linear;
            break;
        case LINEAR_PRIME:
            activeFuncSing = &linearPrime;
            twoArgFunc = false;
            break;
        case SIGMOID:
            activeFunc = &sigmoid;
            break;
        case SIGMOID_PRIME:
            activeFunc = &sigmoidPrime;
            break;
        case TANH:
            activeFunc = &tanhActive;
            break;
        case TANH_PRIME:
            activeFunc = &tanhPrime;
            break;
    }
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            if (!twoArgFunc) {
                matrixSet(new_matrix, i, j, activeFuncSing());
            } else {
                matrixSet(new_matrix, i, j, activeFunc(matrixGet(matrix, i, j)));
            }
        }
    }
    return new_matrix;
}

// PRE: Matrix is square and dilation represents number of zeros to be added between adjacent terms
// POST: Dilated matrix is returned
matrix2d_t *matrixDilate(matrix2d_t *matrix, int dilation) {
    assert(matrix->nCols == matrix->nRows);
    int spaces = (matrix->nRows - 1) * dilation++;
    matrix2d_t *result = matrixCreate(matrix->nRows + spaces, matrix->nCols + spaces);
    for (int i = 0; i < result->nRows; i++) {
        for (int j = 0; j < result->nCols; j++) {
            if (((i % dilation) == 0) && ((j % dilation) == 0)) {
                matrixSet(result, i, j, matrixGet(matrix, i / dilation, j / dilation));
            }
        }
    }
    return result;
}

matrix2d_t *matrixElementWise(matrix2d_t *matrix, double (*func)(double)) {
    matrix2d_t *newMatrix = matrixCreate(matrix->nRows, matrix->nCols);
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            matrixSet(newMatrix, i, j, func(matrixGet(matrix, i, j)));
        }
    }
    return newMatrix;
}

//PRE: Matrix is square
//POST: All elements of matrix are sqrt of original
matrix2d_t *matrixSquareRoot(matrix2d_t *matrix) {
    return matrixElementWise(matrix, sqrt);
}

// PRE: Pointer to initialised input matrix, kernel matrix, stride and padding
// POST: Convolution applied to matrix by multiply input matrix with kernel and summing resulting values
matrix2d_t *matrixConvolution(matrix2d_t *matrix, matrix2d_t *kernel, int stride, int padding) {
    int dimension = ((matrix->nCols - kernel->nCols + 2 * padding) / stride) + 1;
    matrix2d_t *resultMatrix = matrixCreate(dimension, dimension);
    int result = 0;
    int valueInMatrix = 0;
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            result = 0;
            for (int k = 0; k < kernel->nRows; k++) {
                for (int l = 0; l < kernel->nCols; l++) {
                    if (i + k < padding || j + l < padding || i + k >= matrix->nCols + padding || j + l >= matrix->nCols + padding) {
                        //checks if current square is in padding
                        valueInMatrix = 0;
                    } else {
                        valueInMatrix = matrixGet(matrix, stride * (i + k) - padding, stride * (j + l) - padding);
                    }
                    result += valueInMatrix * matrixGet(kernel, k, l);
                }
            }
            matrixSet(resultMatrix, i, j, result);
        }
    }
    return resultMatrix;
}

//PRE: Pointer to initialised input matric, kernel matrix, stride and padding
//POST: Returns deconvoluted matrix
matrix2d_t *matrixDeconvolution(matrix2d_t *matrix, matrix2d_t *kernel, int stride, int padding) {
    int numOfZeros = stride - 1;                                                 //number of zeros to insert between adjacent terms in matrix
    int zeroPadd = kernel->nCols - padding - 1;                                  // amount of padding to matrix

    return matrixConvolution(matrixDilate(matrix, numOfZeros), kernel, 1, zeroPadd);
}

// PRE: Initialised input matrices and kernels.
// POST: 3D convolution applied.
matrix3d_t *matrix3DConvolution(matrix3d_t *inputs, matrix3d_t *kernels, int stride, int padding) {
    assert(inputs);
    assert(kernels);

    int dimension = ((inputs->nCols - kernels->nCols + 2 * padding) / stride) + 1;

    matrix3d_t *results = malloc(sizeof(matrix3d_t));
    results->data = malloc(sizeof(double) * dimension * dimension * kernels->nDepth);
    matrix2d_t *input, *kernel;
    for (int i = 0; i < inputs->nDepth; i++) {
        input = matrixCreate(inputs->nRows, inputs->nCols);
        input->data = inputs->data[0];

        kernel = matrixCreate(kernels->nRows, kernels->nCols);
        kernel->data = kernels->data[0];
        
        results->data[i] = matrixConvolution(input, kernel, stride, padding)->data;
        for (int j = 1; j < kernels->nDepth; j++) {
            matrix2d_t *matConv = matrixConvolution(input, kernel, stride, padding);
            matrix2d_t *tmpResult = matrixCreate(matConv->nRows, matConv->nCols);
            tmpResult->data = results->data[i];
            results->data[i] = matrixAdd(tmpResult, matConv)->data;
        }
    }

    return results;
}

/*
 * Return the maximum value of a part of the give matrix,
 * from rowFrom and colFrom (inclusive) to rowTo and colTo (exclusive).
 */
static double matrixGetLocalMax(matrix2d_t *matrix, matrix2d_t *gradient, int rowFrom, int colFrom, int rowTo, int colTo) {
    assert(matrix);
    int nRows = matrix->nRows;
    int nCols = matrix->nCols;
    assert(nRows > 0);
    assert(nCols > 0);
    assert(rowFrom >= 0);
    assert(colFrom >= 0);
    if (rowTo > nRows) {
        rowTo = nRows;
    }
    if (colTo > nCols) {
        colTo = nCols;
    }

    int iMax = rowFrom;
    int jMax = colFrom;
    double max = matrixGet(matrix, rowFrom, colFrom);
    for (int i = rowFrom; i < rowTo; i++) {
        for (int j = colFrom; j < colTo; j++) {
            double value = matrixGet(matrix, i, j);
            if (value > max) {
                max = value;
                iMax = i;
                jMax = j;
            }
        }
    }

    matrixSet(gradient, iMax, jMax, 1.0);
    return max;
}

/*
 * Return the average value of a part of the give matrix,
 * from rowFrom and colFrom (inclusive) to rowTo and colTo (exclusive).
 */
static double matrixGetLocalAvg(matrix2d_t *matrix, matrix2d_t *gradient, int rowFrom, int colFrom, int rowTo, int colTo) {
    assert(matrix);
    int nRows = matrix->nRows;
    int nCols = matrix->nCols;
    assert(nRows > 0);
    assert(nCols > 0);
    assert(rowFrom >= 0);
    assert(colFrom >= 0);
    int elemNum = (rowTo - rowFrom) * (colTo - colFrom);

    if (rowTo > nRows) {
        rowTo = nRows;
    }
    if (colTo > nCols) {
        colTo = nCols;
    }

    double sum = 0;
    for (int i = rowFrom; i < rowTo; i++) {
        for (int j = colFrom; j < colTo; j++) {
            double value = matrixGet(matrix, i, j);
            sum += value;
        }
    }

    double avg = sum / (double) elemNum;

    for (int i = rowFrom; i < rowTo; i++) {
        for (int j = colFrom; j < colTo; j++) {
            matrixSet(gradient, i, j, avg);
        }
    }

    return avg;
}

static matrix2d_t *matrixPooling(matrix2d_t *matrix, matrix2d_t *gradient, int stride, int filterSize, double (*poolingFunc)(matrix2d_t *, matrix2d_t *, int, int, int, int)) {
    assert(matrix);
    assert(stride);

    int nRows = matrix->nRows;
    int nCols = matrix->nCols;
    int newNRows = nRows / stride;
    int newNCols = nCols / stride;

    matrix2d_t *output = matrixCreate(nRows / stride, nCols / stride);
    gradient = matrixCreate(nRows, nCols);

    for (int i = 0; i < newNRows; i++) {
        for (int j = 0; j < newNCols; j++) {
            int oldI = i * stride;
            int oldJ = j * stride;
            matrixSet(output, i, j, poolingFunc(matrix, gradient, oldI, oldJ, oldI + filterSize, oldJ + filterSize));
        }
    }

    return output;
}

matrix2d_t *matrixMaxPooling(matrix2d_t *matrix, matrix2d_t *gradient, int stride, int filterSize) {
    return matrixPooling(matrix, gradient, stride, filterSize, matrixGetLocalMax);
}

matrix2d_t *matrixAveragePooling(matrix2d_t *matrix, matrix2d_t *gradient, int stride, int filterSize) {
    return matrixPooling(matrix, gradient, stride, filterSize, matrixGetLocalAvg);
}

bool areMatrixesEqual(matrix2d_t *matrix1, matrix2d_t *matrix2, double tolerance) {
    if (matrix1->nRows != matrix2->nRows || matrix1->nCols != matrix2->nCols) {
        return false;
    }
    int nRows = matrix1->nRows;
    int nCols = matrix1->nCols;
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            double matrix1Value = matrixGet(matrix1, i, j);
            double matrix2Value = matrixGet(matrix2, i, j);
            if (matrix1Value > matrix2Value + tolerance || matrix1Value < matrix2Value - tolerance) {
                return false;
            }
        }
    }
    return true;
}

matrix2d_t *matrixFlatten(matrix3d_t *matrix) {
    matrix2d_t *flatTranspose = matrixCreate(1, matrix->nRows * matrix->nCols * matrix->nDepth);

    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            for (int k = 0; k < matrix->nDepth; k++) {
                matrixSet(flatTranspose, (i * matrix->nCols) + (j * matrix->nDepth) + k, 0, matrix->data[i][j][k]);
            }
        }
    }
    matrix2d_t *flattened = matrixTranspose(flatTranspose);
    matrixFree(flatTranspose);
    return flattened;
}

matrix3d_t *matrixUnflatten(matrix2d_t *matrix, int nRows, int nCols, int nDepth) {
    matrix3d_t *unflattened = matrix3DCreate(nRows, nCols, nDepth);

    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            for (int k = 0; k < nDepth; k++) {
                unflattened->data[i][j][k] = matrixGet(matrix, (i * nCols) + (j * nDepth) + k, 0);
            }
        }
    }
    return unflattened;
}

double* flatten2d(matrix2d_t *matrix) {
    double *flattened = calloc(matrix->nRows * matrix->nCols, sizeof(double));
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            flattened[(i * matrix->nCols) + j] = matrixGet(matrix, i, j);
        }
    }
    return flattened;
}

void matrixPrint(matrix2d_t *matrix) {
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            printf("%f ", matrixGet(matrix, i, j));
        }
        printf("\n");
    }
}

void matrixFree(matrix2d_t *matrix) {
    int nRows = matrix->nRows;
    for (int i = 0; i < nRows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}
