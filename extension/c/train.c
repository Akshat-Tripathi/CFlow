#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../train.h"
#include "../compiler.h"
#include "../nodes.h"
#include "../layers.h"
#include "../predict.h"
#include "../scheduler.h"
#include "../graphix.h"
#include "../error.h"
#include "../util.h"
#include "../optimisers.h"

// Load batch of data (randomly shuffled)
// Predict the output
// Generate the loss
// Predict on the backprop graph
// Call optimisers on each node which represents a weight or a bias

#include "../testUtils.h"

//PRE: batchSize <= number of inputs / targets
//PRE: all inputs/targets have the same shape
//POST: returns the targets matrix, and sets the input matrix
matrix2d_t **sample(matrix2d_t **inputs, matrix2d_t **targets, 
                 int batchSize, matrix2d_t ***input, 
                 int nInputs, int nTargets) {
    
    int size = inputs[0]->nRows;

    *input = malloc(sizeof(matrix2d_t*) * nInputs);
    matrix2d_t **target = malloc(sizeof(matrix2d_t*) * nTargets);

    for (int i = 0; i < nInputs; i++) {
        (*input)[i] = matrixCreate(batchSize, inputs[i]->nCols);
    }

    for (int i = 0; i < nTargets; i++) {
        target[i] = matrixCreate(batchSize, targets[i]->nCols);
    }

    int index;
    for (int i = 0; i < batchSize; i++) {
        index = rand() % size;
        for (int j = 0; j < nInputs; j++) {
            for (int k = 0; k < inputs[j]->nCols; k++) {
                matrixSet((*input)[j], i, k, matrixGet(inputs[j], index, k));
            }
        }
        for (int j = 0; j < nTargets; j++) {
            for (int k = 0; k < targets[j]->nCols; k++) {
                matrixSet(target[j], i, k, matrixGet(targets[j], index, k));
            }
        }
    }

    return target;
}

//PRE: inputs contains matrices for first layer
//POST: the network would have been trained for 'epochs' epochs
void train(graph_t *graph, matrix2d_t **inputs, matrix2d_t **targets,
           double lRate, int epochs, enum errorFunction func,
           int batchSize, enum optimiser optimiser) {
    
    va_list args;

    double (*loss)(matrix2d_t*, matrix2d_t*);
    matrix2d_t *(*dLoss)(matrix2d_t*, matrix2d_t*);

    void (*opt)(node_t *weight, int nArgs, ...);
    int nArgs;

    switch (optimiser) {
        case SGD: opt = sgd; nArgs = 1; break;
        case MOMENTUM: opt = sgdMomentum; nArgs = 2; break;
        case ADAGRAD: opt = adagrad; nArgs = 2; break;
        case RMSPROP: opt = RMSProp; nArgs = 3; break;
    }


    switch (func) {
        case MSE: 
            loss = meanSquaredError;
            dLoss = dMeanSquaredError;
            break;
        case CSL:
            loss = crossEntropyLoss;
            dLoss = dCrossEntropyLoss;
    }

    int nNodesForward, nNodesBackward;
    node_t **forward = schedule(graph, &nNodesForward);
    graph_t *compiled = compile(graph, "backprop");
    writeGraph(compiled);
    node_t **backward = schedule(compiled, &nNodesBackward);

    int nInputs = 0, nTargets = graph->m;
    matrix2d_t **input, **target;

    for (int i = 0; i < graph->n; i++) 
        nInputs += !graph->entryPoints[i]->content.data->internalNode;

    //Collect all the losspoints in 1 place
    node_t **lossPoints = malloc(sizeof(node_t*) * nTargets);

    for (int i = 0; i < nTargets; i++) {
        for (int j = 0; j < compiled->n; j++) {
            //Check string pointer equality
            if (compiled->entryPoints[j]->name == graph->exitPoints[i]->name) {
                lossPoints[i] = compiled->entryPoints[j];
                lossPoints[i]->content.data->internalNode = false;
                break;
            }
        }
    }

    int inputIdx = 0;
    double error;

    for (int i = 0; i < epochs + 1; i++) {
        //Prime graphs with data
        if (batchSize < nInputs) {
            target = sample(inputs, targets, batchSize, &input, nInputs, nTargets);
        } else {
            target = targets;
            input = inputs;
        }

        for (int j = 0; j < graph->n && inputIdx < nInputs; j++) {
            if (!graph->entryPoints[j]->content.data->internalNode) 
                graph->entryPoints[j]->content.data->data->matrix2d = input[inputIdx++];
        }

        for (int j = 0; j < nTargets; j++) {
            lossPoints[j]->content.data->data->matrix2d = target[j];
        }

        execute(forward, nNodesForward, FORWARD, NULL, 0);
        

        if (epochs == i) {
            for (int j = 0; j < batchSize; j++) {
                printf("hi");
                printf("Input: [%lf, %lf] -> Prediction: %lf\n",
                                        input[0]->data[j][0],
                                        input[0]->data[j][1],
                                        graph->exitPoints[0]->inputs[0]->matrix->matrix2d->data[j][0]);
            }
            break;
        }

        //generate loss

        node_t *graphPoint, *lossPoint;
        
        for (int j = 0; j < nTargets; j++) {
            graphPoint = graph->exitPoints[j];
            lossPoint = lossPoints[j];
            lossPoint->content.data->data->matrix2d = 
                dLoss(lossPoint->content.data->data->matrix2d, graphPoint->inputs[0]->matrix->matrix2d);
            error = 0;
            double temp;
            for (int k = 0; k < batchSize; k++) {
                temp = matrixGet(lossPoint->content.data->data->matrix2d, k, 0);
                error += temp * temp;
            }
            printf("Loss at epoch: %d is %lf\n", i, error / batchSize);
        }

        execute(backward, nNodesBackward, BACKWARD, opt, nArgs, args);
        execute(backward, nNodesBackward, UPDATE, NULL, 0);
    }
}
