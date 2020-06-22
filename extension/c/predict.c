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

    node_t *node;
    for (int i = length - 1; i >= 0; i--) {
        node = nodes[i];
        if (node->isData) {
            switch (mode) {
            case FORWARD:
                node->matrix = matrixClone(node->content.data->data->matrix2d);
                break;
            case BACKWARD:
                if (node->content.data->internalNode) {
                    if (node->matrix) free(node->matrix);
                    node->matrix = matrixClone(node->content.data->data->matrix2d);
                    for (int j = 0; j < node->n; j++) {
                        node->matrix = matrixAdd(node->matrix, 
                                                 node->inputs[j]->matrix);
                        //optimiser(node, nArgs, args);
                    }
                } else {
                   node->matrix = matrixClone(node->content.data->data->matrix2d); 
                }
                break;
            case UPDATE:
                if (node->content.data->internalNode && 'd' == *(node->name)) {
                    free(node->content.data->data);
                    node->content.data->data->matrix2d = matrixClone(node->matrix);
                    free(node->matrix);
                }
            }
        } else {
            if (UPDATE == mode) continue;
            //if (node->matrix) free(node->matrix);
            switch (node->content.operation.funcName) {
                case CONVOLUTION:
                    //stride and padding are stored in a 2 X 1 matrix
                    node->matrix = matrix3DConvolution(node->inputs[0]->matrix, node->inputs[1]->matrix,
                    matrixGet(node->inputs[2]->matrix, 0, 0), matrixGet(node->inputs[2]->matrix, 0, 1));
                    break;
                case DECONVOLUTION:
                    //stride and padding are stored in a 2 X 1 matrix
                    node->matrix = matrixDeconvolution(node->inputs[0]->matrix, node->inputs[1]->matrix,
                    matrixGet(node->inputs[2]->matrix, 0, 0), matrixGet(node->inputs[2]->matrix, 0, 1));
                    break;
                case ADD:
                    node->matrix = matrixAdd(node->inputs[0]->matrix, node->inputs[1]->matrix);
                    break;
                case SUBTRACT:
                    node->matrix = matrixSubtract(node->inputs[0]->matrix, node->inputs[1]->matrix);
                    break;
                case MULTIPLY:
                    node->matrix = matrixMultiplyElementWise(node->inputs[0]->matrix, node->inputs[1]->matrix);
                    break;
                case DOT:
                    node->matrix = matrixDotProduct(node->inputs[0]->matrix, node->inputs[1]->matrix);
                    break;
                case MAX_POOLING:
                    if (mode == BACKWARD) {
                        matrix2d_t* errorMatrix = matrixCreate(node->inputs[0]->poolingMatrixGrad->nRows, 
                                                             node->inputs[0]->poolingMatrixGrad->nCols);
                        for (int i = 0; i < errorMatrix->nRows; i++) {
                            for (int j = 0; j < errorMatrix->nCols; i++) {
                                if (matrixGet(node->inputs[0]->poolingMatrixGrad, i, j) == 1.0) {
                                    matrixSet(errorMatrix, i, j, matrixGet(node->inputs[0]->matrix, i, j));
                                }
                            }
                        }
                        node->matrix = errorMatrix;
                    } else {
                        //stride and filter size are stored in a 2 X 1 matrix
                        node->matrix = matrixMaxPooling(node->inputs[0]->matrix, node->poolingMatrixGrad,
                        matrixGet(node->inputs[1]->matrix, 0, 0), matrixGet(node->inputs[1]->matrix, 0, 1));
                    }
                    break;
                case AVERAGE_POOLING:
                    if (mode == BACKWARD) {
                        matrix2d_t* errorMatrix = matrixCreate(node->inputs[0]->poolingMatrixGrad->nRows, 
                                                             node->inputs[0]->poolingMatrixGrad->nCols);
                        int filterSize = matrixGet(node->inputs[1]->matrix, 0, 1);
                        for (int i = 0; i < errorMatrix->nRows; i++) {
                            for (int j = 0; j < errorMatrix->nCols; i++) {
                                    matrixSet(errorMatrix, i, j, matrixGet(node->inputs[0]->matrix, i, j) / (double) (filterSize * filterSize));
                            }
                        }
                        node->matrix = errorMatrix;
                    } else {
                        //stride and filter size are stored in a 2 X 1 matrix
                        node->matrix = matrixAveragePooling(node->inputs[0]->matrix, node->poolingMatrixGrad,
                        matrixGet(node->inputs[1]->matrix, 0, 0), matrixGet(node->inputs[1]->matrix, 0, 1));
                    }
                    break;
                case ACTIVATION:
                    node->matrix = matrixActiveFunc(node->inputs[0]->matrix, node->content.operation.activationName);
                    break;
                case TRANSPOSE:
                    node->matrix = matrixTranspose(node->inputs[0]->matrix);
                    break;
                default:
                    printf("I haven't programmed that path in yet\n");
                    exit(EXIT_FAILURE); 
            }
        }
        //if (BACKWARD == mode) {
            printf("%s\n", node->name);
            if (node->matrix) if(node->matrix->data) printMatrix(node->matrix);
            printf("\n\n");
        //}
    }
}
