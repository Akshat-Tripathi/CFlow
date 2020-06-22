#include <stdarg.h>

#include "../optimisers.h"

//PRE: Node containing weight matrix stored in node->content->data and gradients stored in node->matrix
//     and fixed learning rate
//POST: Updates passed node with updated weights 
void sgd(node_t *weight, int nArgs, ...) {

    va_list args;

    va_start(args, nArgs);

    double lRate = va_arg(args, double);

    va_end(args);

    matrix2d_t *newWeights = matrixSubtract(weight->content.data->data->matrix2d, 
                                            matrixScalarProduct(weight->matrix->matrix2d, lRate));
    weight->content.data->data->matrix2d = newWeights;
}

//PRE: Node containing weight matrix stored in node->content->data, gradients stored in node->matrix,
//     velocity stored in node->optimiserMatrix, momentum constant, fixed learning rate and matrix representing velocity
//POST: Updates passed node with updated weights, updated velocity matrix
void sgdMomentum(node_t *weight, int nArgs, ...) {

    va_list args;

    va_start(args, nArgs);

    double lRate = va_arg(args, double);
    double momentum = va_arg(args, double); 
    va_end(args);

    matrix2d_t *newVelocity = matrixSubtract(matrixScalarProduct(weight->optimiserMatrix->matrix2d, momentum), 
                                             matrixScalarProduct(weight->matrix->matrix2d, lRate));
    matrix2d_t *newWeight = matrixAdd(weight->content.data->data->matrix2d, newVelocity);

    weight->content.data->data->matrix2d = newWeight;
    matrixFree(weight->optimiserMatrix->matrix2d);
    weight->optimiserMatrix->matrix2d = newVelocity;
}

static double inverse(double a) {
    return 1.0 / a;
}

//PRE: Node containing weight matrix stored in node->content->data and gradients stored in node->matrix,
//     fixed learning rate, delta constant, and a gradient accumulation vector (r)
//     nVals = 2 
//POST: Updates passed node with updated weights, updated r vector
void adagrad(node_t *weight, int nArgs, ...) {
    
    va_list args;

    va_start(args, nArgs);

    double lRate = va_arg(args, double);
    double delta = va_arg(args, double); 
    va_end(args);

    matrix2d_t *newR = matrixAdd(weight->optimiserMatrix->matrix2d,
                               matrixMultiplyElementWise(weight->matrix->matrix2d, weight->matrix->matrix2d));

    //No easy way to add delta to each of r's elements
    matrix2d_t *denomConstant = matrixSquareRoot(weight->optimiserMatrix->matrix2d);
    for (int i = 0; i < denomConstant->nRows; i++) {
        for (int j = 0; j < denomConstant->nCols; j++) {
            matrixSet(denomConstant, i, j, matrixGet(denomConstant, i, j) + delta);
        }
    }
    matrix2d_t *constant = matrixScalarProduct(matrixElementWise(denomConstant, inverse), lRate);
    matrix2d_t *newWeight = matrixSubtract(weight->content.data->data->matrix2d,
                                         matrixMultiplyElementWise(constant, weight->matrix->matrix2d));
    weight->content.data->data->matrix2d = newWeight;
    matrixFree(weight->optimiserMatrix->matrix2d);
    weight->optimiserMatrix->matrix2d = newR;
    matrixFree(denomConstant);
    matrixFree(constant);
}

//PRE: Node containing weight matrix stored in node->content->data and gradients stored in node->matrix,
//     fixed learning rate, delta and decay rate constant, and a gradient accumulation vector (r)
//POST: Updates passed node with updated weights, updated r vector
void RMSProp(node_t *weight, int nArgs, ...) {

    va_list args;

    va_start(args, nArgs);

    double lRate = va_arg(args, double);
    double decayRate = va_arg(args, double);
    double delta = va_arg(args, double); 
    va_end(args);

    matrix2d_t *newR = matrixAdd(matrixScalarProduct(weight->optimiserMatrix->matrix2d, decayRate),
                               matrixScalarProduct(matrixMultiplyElementWise(weight->matrix->matrix2d, weight->matrix->matrix2d), 1.0 - decayRate));

    //No easy way to add delta to each of r's elements
    matrix2d_t *denomConstant = matrixSquareRoot(weight->optimiserMatrix->matrix2d);
    for (int i = 0; i < denomConstant->nRows; i++) {
        for (int j = 0; j < denomConstant->nCols; j++) {
            matrixSet(denomConstant, i, j, matrixGet(denomConstant, i, j) + delta);
        }
    }
    matrix2d_t *constant = matrixScalarProduct(matrixElementWise(denomConstant, inverse), lRate);
    matrix2d_t *newWeight = matrixSubtract(weight->content.data->data->matrix2d, 
                                         matrixMultiplyElementWise(constant, weight->matrix->matrix2d));
    weight->content.data->data->matrix2d = newWeight;
    matrixFree(weight->optimiserMatrix->matrix2d);
    weight->optimiserMatrix->matrix2d = newR;
    matrixFree(denomConstant);
    matrixFree(constant);
}