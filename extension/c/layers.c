#include <stdlib.h>
#include <stdio.h>

#include "../nodes.h"
#include "../matrix.h"
#include "../util.h"
#include "../activation.h"
#include "../layers.h"

matrix2d_t *generateRandMatrix(int nRows, int nCols) {
    matrix2d_t *randMT = matrixCreate(nRows, nCols);   
    matrixRandomise(randMT);
    return randMT;
}

int nDense = 0;
int nConvLayers = 0;
int nLSTM = 0;
int nGates = 0;

//PRE: Number of neurons, input node number of features and a activation function to apply
//POST: Links the input node to a dense layer and returns the output of the dense layer
node_t *denseLayer(node_t *x, int nNeurons,
        enum activationFunction activationFunction, 
        node_t ***entryPoints, int *length) {

    matrix2d_t *weightMT = generateRandMatrix(x->matrix->matrix2d->nCols, nNeurons);
    matrix2d_t *biasMT = matrixCreate(x->matrix->matrix2d->nRows, nNeurons);
    
    char *weightName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *biasName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *dotName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *addName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *funcName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));

    snprintf(weightName, MAX_NODE_NAME_LENGTH, "WEIGHT%d", nDense);
    snprintf(biasName, MAX_NODE_NAME_LENGTH, "BIAS%d", nDense);
    snprintf(dotName, MAX_NODE_NAME_LENGTH, "DOT%d", nDense);
    snprintf(addName, MAX_NODE_NAME_LENGTH, "ADD%d", nDense);
    snprintf(funcName, MAX_NODE_NAME_LENGTH, "FUNC%d", nDense++);

    node_t* weight = nodeInit(weightName, 0, 1, true);
    weight->content.data->data->matrix2d = weightMT;

    node_t* bias = nodeInit(biasName, 0, 1, true);
    bias->content.data->data->matrix2d = biasMT;

    node_t *dotProduct = nodeInit(dotName, 2, 1, false);
    dotProduct->content.operation = (operation_t) {.funcName = DOT};

    node_t *add = nodeInit(addName, 2, 1, false);
    add->content.operation = (operation_t) {.funcName = ADD};

    node_t *activFunc = nodeInit(funcName, 1, 1, false);
    activFunc->content.operation = (operation_t) {.funcName = ACTIVATION, .activationName = activationFunction};

    linkNodes(x, dotProduct);
    linkNodes(weight, dotProduct);
    linkNodes(dotProduct, add);
    linkNodes(bias, add);
    linkNodes(add, activFunc);

    push(entryPoints, length, weight);
    push(entryPoints, length, bias);

    activFunc->matrix->matrix2d = matrixCreate(x->matrix->matrix2d->nRows, nNeurons);

    return activFunc;
}

//TODO: fix this with 3d matrices
//PRE: The input is a square, the kernel is a square and a activation function to apply
//POST: A incomplete convolutional layer that requires links to an input node (x) and a output node that will store result 
node_t *convolutionalLayer(node_t *x, int nChannels, int kernelSize,
                           int nkernels, enum matrixFunction activationFunction,
                           int stride, int padding, node_t ***entryPoints,
                           int *length) {

    matrix2d_t *kernelMT = generateRandMatrix(kernelSize, kernelSize);

    int dimension = ((x->matrix->matrix2d->nCols - kernelSize + 2 * padding) / stride) + 1;

    matrix2d_t *biasMT = matrixCreate(dimension, dimension);
    
    char *kernelName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *biasName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *convName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *addName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *funcName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));

    snprintf(kernelName, MAX_NODE_NAME_LENGTH, "KERNEL%d", nConvLayers);
    snprintf(biasName, MAX_NODE_NAME_LENGTH, "CBIAS%d", nConvLayers);
    snprintf(convName, MAX_NODE_NAME_LENGTH, "CONV%d", nConvLayers);
    snprintf(addName, MAX_NODE_NAME_LENGTH, "CADD%d", nConvLayers);
    snprintf(funcName, MAX_NODE_NAME_LENGTH, "CFUNC%d", nConvLayers++);

    node_t* kernel = nodeInit(kernelName, 0, 1, true);
    kernel->content.data->data->matrix2d = kernelMT;

    node_t* bias = nodeInit(biasName, 0, 1, true);
    bias->content.data->data->matrix2d = biasMT;

    node_t *conv = nodeInit(convName, 3, 1, false);
    conv->content.operation = (operation_t) {.funcName = CONVOLUTION};

    node_t *add = nodeInit(addName, 2, 1, false);
    add->content.operation = (operation_t) {.funcName = ADD};

    node_t *activFunc = nodeInit(funcName, 1, 1, false);
    activFunc->content.operation = (operation_t) {.funcName = ACTIVATION, .activationName = activationFunction};

    node_t *data = nodeInit("config", 0, 1, true);
    data->content.data->data->matrix2d = matrixCreate(1, 2);
    matrixSet(data->content.data->data->matrix2d, 0, 0, stride);
    matrixSet(data->content.data->data->matrix2d, 0, 1, padding);

    linkNodes(x, conv);
    linkNodes(kernel, conv);
    linkNodes(data, conv);
    linkNodes(conv, add);
    linkNodes(bias, add);
    linkNodes(add, activFunc);

    push(entryPoints, length, kernel);
    push(entryPoints, length, bias);

    activFunc->matrix->matrix2d = matrixCreate(dimension, dimension);

    return activFunc;
}

//From: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9

//PRE: Accepts an input and the previous cell's output - both must have 4 outputs
//PRE: Accepts the activation func
//POST: Links the input and previous output to the gate and returns the gate
static node_t *LSTMGate(node_t *x, node_t *prevOutput, 
        enum activationFunction activationFunction,
        node_t ***entryPoints, int *length) {

    //activationFunction(dot(x, W) + dot(prev, U) + b)

    int nNeurons  = prevOutput->matrix->matrix2d->nCols;
    //int batchSize = prevOutput->matrix->nRows;

    matrix2d_t *weightMTW = generateRandMatrix(x->matrix->matrix2d->nCols, nNeurons);
    matrix2d_t *weightMTU = generateRandMatrix(nNeurons, nNeurons);
    matrix2d_t *biasMT = matrixCreate(x->matrix->matrix2d->nRows, nNeurons);
    
    char *weightWName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *weightUName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *biasName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *dotWName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *dotUName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *addName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *addBName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *gateName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));

    snprintf(weightWName, MAX_NODE_NAME_LENGTH, "WEIGHTW%d", nGates);
    snprintf(weightUName, MAX_NODE_NAME_LENGTH, "WEIGHTU%d", nGates);
    snprintf(biasName, MAX_NODE_NAME_LENGTH, "BIAS%d", nGates);
    snprintf(dotWName, MAX_NODE_NAME_LENGTH, "DOTW%d", nGates);
    snprintf(dotUName, MAX_NODE_NAME_LENGTH, "DOTU%d", nGates);
    snprintf(addName, MAX_NODE_NAME_LENGTH, "ADD%d", nGates);
    snprintf(addBName, MAX_NODE_NAME_LENGTH, "ADDB%d", nGates);
    snprintf(gateName, MAX_NODE_NAME_LENGTH, "FUNC%d", nGates++);

    node_t *weightW = nodeInit(weightWName, 0, 1, true);
    weightW->content.data->data->matrix2d = weightMTW;

    node_t *weightU = nodeInit(weightUName, 0, 1, true);
    weightU->content.data->data->matrix2d = weightMTU;

    node_t *bias = nodeInit(biasName, 0, 1, true);
    bias->content.data->data->matrix2d = biasMT;

    node_t *dotProductW = nodeInit(dotWName, 2, 1, false);
    dotProductW->content.operation = (operation_t) {.funcName = DOT};

    node_t *dotProductU = nodeInit(dotUName, 2, 1, false);
    dotProductU->content.operation = (operation_t) {.funcName = DOT};

    node_t *add = nodeInit(addName, 2, 1, false);
    add->content.operation = (operation_t) {.funcName = ADD};

    node_t *addB = nodeInit(addBName, 2, 1, false);
    addB->content.operation = (operation_t) {.funcName = ADD};

    node_t *gate = nodeInit(gateName, 1, 1, false);
    gate->content.operation = (operation_t) {.funcName = ACTIVATION, .activationName = activationFunction};

    linkNodes(x, dotProductW);
    linkNodes(weightW, dotProductW);

    linkNodes(prevOutput, dotProductU);
    linkNodes(weightU, dotProductU);

    linkNodes(dotProductW, add);
    linkNodes(dotProductU, add);

    linkNodes(bias, addB);
    linkNodes(add, addB);

    linkNodes(addB, gate);

    push(entryPoints, length, weightW);
    push(entryPoints, length, weightU);
    push(entryPoints, length, bias);

    return gate;
}

//TODO move to test
node_t *genDouble(node_t *x, node_t *prevOutput,
        enum activationFunction activationFunction,
        node_t ***entryPoints, int *length) {

    node_t *activationGate = LSTMGate(x, prevOutput, TANH, entryPoints, length);
    node_t *inputGate = LSTMGate(x, prevOutput, activationFunction, entryPoints, length);

    char *mult1Name = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));

    snprintf(mult1Name, MAX_NODE_NAME_LENGTH, "MULTIPLYX%d", nLSTM);

    node_t *mult1 = nodeInit(mult1Name, 2, 1, false);
    mult1->content.operation = (operation_t) {.funcName = MULTIPLY};
    
    linkNodes(activationGate, mult1);
    linkNodes(inputGate, mult1);
    
    return mult1;
}

//PRE: Accepts an input, the previous cell's output (both must have 4 outputs)
//PRE: And the previous cell's state and a null pointer for the next cell
//POST: Returns (in this order) the output and state of this cell
static node_t *LSTMCell(node_t *x, node_t *prevOutput, node_t *prevState,
                 enum activationFunction func, node_t **nextState,
                 node_t ***entryPoints, int *length) {
    
    push(entryPoints, length, prevOutput);
    push(entryPoints, length, prevState);               

    node_t *activationGate = LSTMGate(x, prevOutput, TANH, entryPoints, length);
    node_t *inputGate = LSTMGate(x, prevOutput, func, entryPoints, length);
    node_t *forgetGate = LSTMGate(x, prevOutput, func, entryPoints, length);
    //Not the same as the cell's output
    node_t *outputGate = LSTMGate(x, prevOutput, func, entryPoints, length);

    char *mult1Name = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *mult2Name = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *stateName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));

    char *tanhName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char *outputName = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));

    snprintf(mult1Name, MAX_NODE_NAME_LENGTH, "MULTIPLY1%d", nLSTM);
    snprintf(mult2Name, MAX_NODE_NAME_LENGTH, "MULTIPLY2%d", nLSTM);
    snprintf(stateName, MAX_NODE_NAME_LENGTH, "STATE%d", nLSTM);
    snprintf(tanhName, MAX_NODE_NAME_LENGTH, "TANH%d", nLSTM);
    snprintf(outputName, MAX_NODE_NAME_LENGTH, "OUTPUT%d", nLSTM++);

    node_t *mult1 = nodeInit(mult1Name, 2, 1, false);
    mult1->content.operation = (operation_t) {.funcName = MULTIPLY};

    node_t *mult2 = nodeInit(mult2Name, 2, 1, false);
    mult2->content.operation = (operation_t) {.funcName = MULTIPLY};

    node_t *state = nodeInit(stateName, 2, 2, false);
    state->content.operation = (operation_t) {.funcName = ADD};

    node_t *tanh = nodeInit(tanhName, 2, 1, false);
    tanh->content.operation = (operation_t) {.funcName = ACTIVATION, .activationName = TANH};

    node_t *output = nodeInit(outputName, 2, 4, false);
    output->content.operation = (operation_t) {.funcName = MULTIPLY};
    
    linkNodes(activationGate, mult1);
    linkNodes(inputGate, mult1);

    linkNodes(prevState, mult2);
    linkNodes(forgetGate, mult2);

    linkNodes(mult1, state);
    linkNodes(mult2, state);

    linkNodes(state, tanh);
    linkNodes(tanh, output);
    linkNodes(outputGate, output);

    state->matrix->matrix2d = matrixCreate(prevState->matrix->matrix2d->nRows,
                                 prevState->matrix->matrix2d->nCols);

    *nextState = state;

    output->matrix->matrix2d = matrixCreate(prevOutput->matrix->matrix2d->nRows,
                                  prevOutput->matrix->matrix2d->nCols);

    return output;
}

static node_t *clone(node_t *node) {
    node_t *cln = nodeInit(node->name, node->n, node->m, node->isData);
    cln->content = node->content;
    cln->matrix->matrix2d = matrixCreate(node->matrix->matrix2d->nRows, node->matrix->matrix2d->nCols);
    return cln; 
}

//PRE: inputs has timeSteps nodes, timeSteps > 0; all inputs have the same shape
graph_t *LSTM(node_t **inputs, int timeSteps, enum activationFunction func, int nNeurons) {
    node_t **entryPoints = NULL; //contains weights and biases
    int length = 0;

    node_t *prevOutput = nodeInit("OUTPUT", 0, 4, true);
    prevOutput->matrix->matrix2d = matrixCreate(inputs[0]->matrix->matrix2d->nRows, nNeurons);
    prevOutput->content.data->data->matrix2d = prevOutput->matrix->matrix2d;
    


    node_t *prevState = nodeInit("STATE", 0, 1, true);
    prevState->matrix->matrix2d = matrixCreate(inputs[0]->matrix->matrix2d->nRows, nNeurons);
    prevState->content.data->data->matrix2d = prevState->matrix->matrix2d;
    
    node_t *nextState;

    node_t *output = LSTMCell(inputs[0], clone(prevOutput), clone(prevState), func, 
                              &nextState, &entryPoints, &length);
    
    push(&entryPoints, &length, inputs[0]);
    
    node_t **linkPoints;
    int nLink;
    for (int t = 1; t < timeSteps; t++) {
        push(&entryPoints, &length, inputs[t]);
        linkPoints = NULL;
        nLink = 0;
        prevOutput = output;
        prevState = nextState;
        output = LSTMCell(inputs[t], prevOutput, prevState, func,
                          &nextState, &linkPoints, &nLink);
        
        //Link all points which aren't state or output
        entryPoints = realloc(entryPoints, sizeof(node_t*) * (length + nLink - 2));
        for (int i = 2; i < nLink; i++) {
            entryPoints[length + i - 2] = linkPoints[i];
        }
        length += nLink - 2;
    }

    node_t *y = nodeInit("y", 1, 0, true);
    y->content.data->internalNode = false;
    y->inputs[0] = output;
    output->outputs[0] = y;
    output->m = 1;
    nextState->m = 1; 

    node_t **exitPoints = malloc(sizeof(node_t*));
    exitPoints[0] = output;

    return graphInit("LSTM", length, entryPoints, 1, exitPoints);
}