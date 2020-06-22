#include <stdlib.h>
#include <stdarg.h>

#include "../predict.h"
#include "../data.h"
#include "../matrix.h"
#include "../nodes.h"
#include "../util.h"
#include "../testUtils.h"

// PRE: A topological sort of the graph (reversed order) and it's length
// POST: Each node's matrix value is set depending on the operation
void execute(node_t **nodes, int length, enum executionMode mode,
             void (*optimiser)(node_t *, int nArgs, ...), int nArgs, ...) {

    va_list args;

    node_t *node;
    for (int i = length - 1; i >= 0; i--) {
        node = nodes[i];
        if (node->isData) {
            switch (mode) {
            case FORWARD:
                node->matrix->matrix2d = matrixClone(node->content.data->data->matrix2d);
                break;
            case BACKWARD:
                if (node->content.data->internalNode) {
                    if (node->matrix->matrix2d) free(node->matrix->matrix2d);
                    node->matrix->matrix2d = matrixClone(node->content.data->data->matrix2d);
                    for (int j = 0; j < node->n; j++) {
                        node->matrix->matrix2d = matrixAdd(node->matrix->matrix2d, 
                                                 node->inputs[j]->matrix->matrix2d);
                        optimiser(node, nArgs, args);
                    }
                } else {
                   node->matrix->matrix2d = matrixClone(node->content.data->data->matrix2d); 
                }
                break;
            case UPDATE:
                if (node->content.data->internalNode && 'd' == *(node->name)) {
                    free(node->content.data->data->matrix2d);
                    node->content.data->data->matrix2d = matrixClone(node->matrix->matrix2d);
                    free(node->matrix->matrix2d);
                }
            }
        } else {
            if (UPDATE == mode) continue;
            //if (node->matrix->matrix2d) free(node->matrix->matrix2d);
            switch (node->content.operation.funcName) {
                case CONVOLUTION:
                    //stride and padding are stored in a 2 X 1 matrix
                    node->matrix->matrix3d = matrix3DConvolution(node->inputs[0]->matrix->matrix3d, node->inputs[1]->matrix->matrix3d,
                    matrixGet(node->inputs[2]->matrix->matrix2d, 0, 0), matrixGet(node->inputs[2]->matrix->matrix2d, 0, 1));
                    break;
                case DECONVOLUTION:
                    //stride and padding are stored in a 2 X 1 matrix
                    node->matrix->matrix2d = matrixDeconvolution(node->inputs[0]->matrix->matrix2d, node->inputs[1]->matrix->matrix2d,
                    matrixGet(node->inputs[2]->matrix->matrix2d, 0, 0), matrixGet(node->inputs[2]->matrix->matrix2d, 0, 1));
                    break;
                case ADD:
                    node->matrix->matrix2d = matrixAdd(node->inputs[0]->matrix->matrix2d, node->inputs[1]->matrix->matrix2d);
                    break;
                case SUBTRACT:
                    node->matrix->matrix2d = matrixSubtract(node->inputs[0]->matrix->matrix2d, node->inputs[1]->matrix->matrix2d);
                    break;
                case MULTIPLY:
                    node->matrix->matrix2d = matrixMultiplyElementWise(node->inputs[0]->matrix->matrix2d, node->inputs[1]->matrix->matrix2d);
                    break;
                case DOT:
                    node->matrix->matrix2d = matrixDotProduct(node->inputs[0]->matrix->matrix2d, node->inputs[1]->matrix->matrix2d);
                    break;
                case MAX_POOLING:
                    if (mode == BACKWARD) {
                        matrix2d_t* errorMatrix = matrixCreate(node->inputs[0]->poolingMatrixGrad->matrix2d->nRows, 
                                                             node->inputs[0]->poolingMatrixGrad->matrix2d->nCols);
                        for (int i = 0; i < errorMatrix->nRows; i++) {
                            for (int j = 0; j < errorMatrix->nCols; i++) {
                                if (matrixGet(node->inputs[0]->poolingMatrixGrad->matrix2d, i, j) == 1.0) {
                                    matrixSet(errorMatrix, i, j, matrixGet(node->inputs[0]->matrix->matrix2d, i, j));
                                }
                            }
                        }
                        node->matrix->matrix2d = errorMatrix;
                    } else {
                        //stride and filter size are stored in a 2 X 1 matrix
                        node->matrix->matrix2d = matrixMaxPooling(node->inputs[0]->matrix->matrix2d, node->poolingMatrixGrad->matrix2d,
                        matrixGet(node->inputs[1]->matrix->matrix2d, 0, 0), matrixGet(node->inputs[1]->matrix->matrix2d, 0, 1));
                    }
                    break;
                case AVERAGE_POOLING:
                    if (mode == BACKWARD) {
                        matrix2d_t* errorMatrix = matrixCreate(node->inputs[0]->poolingMatrixGrad->matrix2d->nRows, 
                                                             node->inputs[0]->poolingMatrixGrad->matrix2d->nCols);
                        int filterSize = matrixGet(node->inputs[1]->matrix->matrix2d, 0, 1);
                        for (int i = 0; i < errorMatrix->nRows; i++) {
                            for (int j = 0; j < errorMatrix->nCols; i++) {
                                    matrixSet(errorMatrix, i, j, matrixGet(node->inputs[0]->matrix->matrix2d, i, j) / (double) (filterSize * filterSize));
                            }
                        }
                        node->matrix->matrix2d = errorMatrix;
                    } else {
                        //stride and filter size are stored in a 2 X 1 matrix
                        node->matrix->matrix2d = matrixAveragePooling(node->inputs[0]->matrix->matrix2d, node->poolingMatrixGrad->matrix2d,
                        matrixGet(node->inputs[1]->matrix->matrix2d, 0, 0), matrixGet(node->inputs[1]->matrix->matrix2d, 0, 1));
                    }
                    break;
                case ACTIVATION:
                    node->matrix->matrix2d = matrixActiveFunc(node->inputs[0]->matrix->matrix2d, node->content.operation.activationName);
                    break;
                case TRANSPOSE:
                    node->matrix->matrix2d = matrixTranspose(node->inputs[0]->matrix->matrix2d);
                    break;
                case FLATTEN:
                    if (mode == FORWARD) {
                        node->matrix->matrix2d = matrixFlatten(node->inputs[0]->matrix->matrix3d);
                    } else {
                        double *config = node->inputs[1]->matrix->matrix2d->data[0];
                        node->matrix->matrix3d = matrixUnflatten(node->inputs[0]->matrix->matrix2d, config[0], config[1], config[2]);
                    }
                default:
                    printf("I haven't programmed that path in yet\n");
                    exit(EXIT_FAILURE); 
            }
        }
        /*if (FORWARD == mode) {
            printf("%s\n", node->name);
            if (node->matrix->matrix2d) if(node->matrix->matrix2d->data) printMatrix(node->matrix->matrix2d);
            printf("\n\n");
        }*/
    }
}
