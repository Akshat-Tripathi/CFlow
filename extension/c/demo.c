#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#include "../compiler.h"
#include "../nodes.h"
#include "../layers.h"
#include "../graphix.h"
#include "../file.h"
#include "../error.h"
#include "../optimisers.h"
#include "../readCSV.h"
#include "../testUtils.h"
#include "../train.h"
#include "../util.h"

//Graph used
//https://www.researchgate.net/profile/Ritu_Kundu/publication/276851762/figure/fig5/AS:669689269211149@1536677772684/A-directed-acyclic-graph-G-V-E-The-set-of-nodes-V-v1-v2-v15-Note.png
graph_t* genGraphOne() {
    node_t *v1 = nodeInit("v1", 0, 2, true);
    node_t *v2 = nodeInit("v2", 1, 1, true);
    node_t *v3 = nodeInit("v3", 2, 3, true);
    node_t *v4 = nodeInit("v4", 1, 1, true);
    node_t *v5 = nodeInit("v5", 1, 2, true);
    node_t *v6 = nodeInit("v6", 1, 2, true);
    node_t *v7 = nodeInit("v7", 2, 1, true);
    node_t *v8 = nodeInit("v8", 3, 2, true);
    node_t *v9 = nodeInit("v9", 1, 1, true);
    node_t *v10 = nodeInit("v10", 2, 1, true);
    node_t *v11 = nodeInit("v11", 1, 1, true);
    node_t *v12 = nodeInit("v12", 1, 1, true);
    node_t *v13 = nodeInit("v13", 1, 2, true);
    node_t *v14 = nodeInit("v14", 3, 0, true);
    node_t *v15 = nodeInit("v15", 1, 1, true);

    linkNodes(v1, v2);
    linkNodes(v1, v3);
    linkNodes(v2, v3);
    linkNodes(v3, v4);
    linkNodes(v3, v5);
    linkNodes(v3, v11);
    linkNodes(v4, v8);
    linkNodes(v5, v6);
    linkNodes(v5, v9);
    linkNodes(v6, v7);
    linkNodes(v6, v10);
    linkNodes(v9, v10);
    linkNodes(v10, v7);
    linkNodes(v7, v8);
    linkNodes(v8, v13);
    linkNodes(v8, v14);
    linkNodes(v11, v12);
    linkNodes(v12, v8);
    linkNodes(v13, v14);
    linkNodes(v13, v15);
    linkNodes(v15, v14);

    node_t** bufferEntryPoints = calloc(1, sizeof(node_t*));
    bufferEntryPoints[0] = v1;
    node_t** bufferExitPoints = calloc(1, sizeof(node_t*));
    bufferExitPoints[0] = v14;

    return graphInit("DAG_1", 1, bufferEntryPoints, 1, bufferExitPoints);
}

//Uses this graph
//https://miro.medium.com/max/4000/1*Fi1AZPZLrGf-6wM_wTSPQw.png
graph_t *genGraphTwo() {
    matrix2d_t *matrix = matrixCreate(20, 10);

    node_t *A = nodeInit("A", 0, 1, true);
    A->content.data->data->matrix2d = matrix;
    node_t *B = nodeInit("B", 1, 3, true);
    B->content.data->data->matrix2d = matrix;
    node_t *C = nodeInit("C", 1, 1, true);
    C->content.data->data->matrix2d = matrix;
    node_t *D = nodeInit("D", 2, 1, true);
    D->content.data->data->matrix2d = matrix;
    node_t *E = nodeInit("E", 3, 1, true);
    E->content.data->data->matrix2d = matrix;
    node_t *F = nodeInit("F", 1, 0, true);
    F->content.data->data->matrix2d = matrix;
    node_t *G = nodeInit("G", 0, 1, true);
    G->content.data->data->matrix2d = matrix;

    linkNodes(A, B);
    linkNodes(G, D);
    linkNodes(B, C);
    linkNodes(B, D);
    linkNodes(C, E);
    linkNodes(B, E);
    linkNodes(D, E);
    linkNodes(E, F);

    node_t **bufferEntryPoints = calloc(2, sizeof(node_t *));
    bufferEntryPoints[0] = A;
    bufferEntryPoints[1] = G;
    node_t **bufferExitPoints = calloc(1, sizeof(node_t*));
    bufferExitPoints[0] = F;

    return graphInit("DAG_2", 2, bufferEntryPoints, 1, bufferExitPoints);
}

//Uses this graph
//https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Directed_acyclic_graph_2.svg/1200px-Directed_acyclic_graph_2.svg.png
graph_t *genGraphThree() {
    matrix2d_t *matrix = matrixCreate(20, 10);

    node_t *n5 = nodeInit("5", 0, 1, true);
    n5->content.data->data->matrix2d = matrix;
    node_t *n7 = nodeInit("7", 0, 2, true);
    n7->content.data->data->matrix2d = matrix;
    node_t *n3 = nodeInit("3", 0, 2, true);
    n3->content.data->data->matrix2d = matrix;
    node_t *n11 = nodeInit("11", 2, 3, true);
    n11->content.data->data->matrix2d = matrix;
    node_t *n8 = nodeInit("8", 2, 1, true);
    n8->content.data->data->matrix2d = matrix;
    node_t *n2 = nodeInit("2", 1, 0, true);
    n2->content.data->data->matrix2d = matrix;
    node_t *n9 = nodeInit("9", 2, 0, true);
    n9->content.data->data->matrix2d = matrix;
    node_t *n10 = nodeInit("10", 2, 0, true);
    n10->content.data->data->matrix2d = matrix;

    linkNodes(n5, n11);
    linkNodes(n7, n11);
    linkNodes(n7, n8);
    linkNodes(n3, n8);
    linkNodes(n11, n2);
    linkNodes(n11, n9);
    linkNodes(n11, n10);
    linkNodes(n8, n9);
    linkNodes(n3, n10);

    node_t **bufferEntryPoints = calloc(3, sizeof(node_t *));
    bufferEntryPoints[0] = n5;
    bufferEntryPoints[1] = n7;
    bufferEntryPoints[2] = n3;
    node_t **bufferExitPoints = calloc(3, sizeof(node_t*));
    bufferExitPoints[0] = n2;
    bufferExitPoints[1] = n9;
    bufferExitPoints[2] = n10;

    return graphInit("DAG_3", 3, bufferEntryPoints, 3, bufferExitPoints);

}

//PRE: inputs has 2 columns
//POST: each row of outputs is the result of XOR on the input rows 
matrix2d_t **xor(matrix2d_t ***inputs) {
    const int n = 4;
    *inputs = malloc(sizeof(matrix2d_t*));
    matrix2d_t **targets = malloc(sizeof(matrix2d_t*));
    int first, second;

    (*inputs)[0] = matrixCreate(4, 2);
    targets[0] = matrixCreate(4, 1);

    for (int i = 0; i < n; i++) {
        first = i & 2;  //Get 1st digit
        second = i & 1; //Get 2nd digit
        matrixSet((*inputs)[0], i, 0, !first);
        matrixSet((*inputs)[0], i, 1, !second);
        matrixSet(targets[0], i, 0, !first ^ !second);
    }

    return targets;
}

void trainXOR(void) {
    //Create xor net
    node_t *x = nodeInit("x", 0, 1, true);
    x->content.data->internalNode = false;

    int batchSize = 4;
    
    x->matrix->matrix2d = matrixCreate(batchSize, 2);

    node_t **entryPoints = NULL;
    int n = 0;

    push(&entryPoints, &n, x);

    node_t *layer1 = denseLayer(x, 2, SIGMOID, &entryPoints, &n);
    node_t *layer2 = denseLayer(layer1, 1, SIGMOID, &entryPoints, &n);
    node_t *y = nodeInit("y", 1, 0, true);

    layer2->outputs[0] = y;
    y->inputs[0] = layer2;
    y->content.data->internalNode = false;

    node_t **exitPoints = malloc(sizeof(node_t*));
    exitPoints[0] = y;

    graph_t *network = graphInit("xor", n, entryPoints, 1, exitPoints);
    writeGraph(network);

    matrix2d_t **inputs;
    matrix2d_t **targets = xor(&inputs);

    train(network, inputs, targets, 0.1, 100, MSE, batchSize, SGD);

    //freeGraph(network);
    free(inputs[0]);
    free(targets[0]);
    free(inputs);
    free(targets);

}

void trainMNIST() {
    node_t *x = nodeInit("x", 0, 1, true);
    x->content.data->internalNode = false;

    int nInstances = 60000;

    // Load training data
    x->matrix->matrix3d = matrix3DCreate(nInstances, 28, 28);
    csvDataPack_t trainingData = readCSV("data/mnist_train.csv", nInstances);
    matrix2d_t** matrixData = trainingData.matrixInputs;
    for (int i = 0; i < nInstances; i++) {
        x->matrix->matrix3d->data[i] = matrixData[i]->data;
    }


    node_t **entryPoints = NULL;
    int n = 0;

    push(&entryPoints, &n, x);


    // Add sequential model
    node_t *layer1 = convolutionalLayer(x, 1, 3, 32, RELU, 1, 0, &entryPoints, &n);

    // Max Pooling Layer
    matrix3d_t *toMaxPool = layer1->matrix->matrix3d;
    matrix2d_t **maxPoolGradients = calloc(toMaxPool->nRows, sizeof(matrix2d_t*));

    for (int i = 0; i < toMaxPool->nRows; i++) {
        matrix2d_t *tmp = matrixCreate(toMaxPool->nCols, toMaxPool->nDepth);
        tmp->data = toMaxPool->data[i];

        matrix2d_t* maxPooledTmp = matrixMaxPooling(tmp, maxPoolGradients[i], 1, 2);
        toMaxPool->data[i] = maxPooledTmp->data;
    }

    layer1->matrix->matrix3d = toMaxPool;

    // Flatten layer
    layer1->matrix->matrix2d = matrixFlatten(toMaxPool);

    node_t *layer2 = denseLayer(layer1, 128, RELU, &entryPoints, &n);


    node_t *layer3 = denseLayer(layer2, 10, SIGMOID, &entryPoints, &n);

    // Output layer
    node_t *y = nodeInit("y", 1, 0, true);
    y->content.data->internalNode = false;

    node_t **exitPoints = malloc(sizeof(node_t*));
    exitPoints[0] = y;

    graph_t *network = graphInit("mnist", n, entryPoints, 1, exitPoints);

    // Convert labels into CFlow format
    matrix2d_t *labelMTTransposed = matrixCreate(1, nInstances);
    labelMTTransposed->data[0] = trainingData.labels;

    matrix2d_t **inputs;
    matrix2d_t **targets = calloc(1, sizeof(matrix2d_t*));
    targets[0] = matrixTranspose(labelMTTransposed);

    linkNodes(layer3, y);
    //writeGraph(network);
    // Don't know if batch size of 100 is right
    train(network, inputs, targets, 0.1, 100, CSL, 100, SGD);
}

void trainMNISTSimple() {
    node_t *x = nodeInit("x", 0, 1, true);
    x->content.data->internalNode = false;

    int nInstances = 60000;

    int batchSize = 1;
    x->matrix->matrix2d = matrixCreate(batchSize, 784);

    // Loading training data
    csvDataPack_t trainingData = readCSV("data/mnist_train.csv", nInstances);
    matrix2d_t** inputs = calloc(1, sizeof(matrix2d_t*));
    inputs[0] = matrixCreate(nInstances, 784);
    matrix2d_t** matrixData = trainingData.matrixInputs;
    for (int i = 0; i < nInstances; i++) {
        inputs[0]->data[i] = flatten2d(matrixData[i]);
    }


    node_t **entryPoints = NULL;
    int n = 0;

    push(&entryPoints, &n, x);

    //Add sequential model
    node_t *layer1 = denseLayer(x, 64, RELU, &entryPoints, &n);
    node_t *layer2 = denseLayer(layer1, 64, RELU, &entryPoints, &n);
    node_t *layer3 = denseLayer(layer2, 10, SIGMOID, &entryPoints, &n);

    node_t *y = nodeInit("y", 1, 0, true);
    y->content.data->internalNode = false;

    linkNodes(layer3, y);

    node_t **exitPoints = malloc(sizeof(node_t*));
    exitPoints[0] = y;
    graph_t *network = graphInit("mnist", n, entryPoints, 1, exitPoints);

    // Convert labels into CFlow format
    matrix2d_t *labelMTTransposed = matrixCreate(1, nInstances);
    labelMTTransposed->data[0] = trainingData.labels;

    matrix2d_t **targets = calloc(1, sizeof(matrix2d_t*));
    targets[0] = matrixTranspose(labelMTTransposed);

    // Don't know if batch size of 100 is right
    train(network, inputs, targets, 0.1, 5, CSL, batchSize, SGD);
    
}

int main(void) {

    /*graph_t *graphOne = genGraphOne();
    graphFileWrite("DAG_1_SAVE", graphOne);
    graph_t *graphOneRead = graphFileRead("DAG_1_SAVE");
    writeGraph(graphOneRead);

    graph_t *graphTwo = genGraphTwo();
    graphFileWrite("DAG_2_SAVE", graphTwo);
    graph_t *graphTwoRead = graphFileRead("DAG_2_SAVE");
    writeGraph(graphTwoRead);

    graph_t *graphThree = genGraphThree();
    graphFileWrite("DAG_3_SAVE", graphThree);
    graph_t *graphThreeRead = graphFileRead("DAG_3_SAVE");
    writeGraph(graphThreeRead);
    
    node_t **inputs = malloc(sizeof(node_t*) * 4);
    inputs[0] = nodeInit("x0", 0, 4, true);
    inputs[0]->matrix = matrixCreate(4, 6);
    inputs[1] = nodeInit("x1", 0, 4, true);
    inputs[1]->matrix = matrixCreate(4, 6);
    inputs[2] = nodeInit("x2", 0, 4, true);
    inputs[2]->matrix = matrixCreate(4, 6);
    inputs[3] = nodeInit("x3", 0, 4, true);
    inputs[3]->matrix = matrixCreate(4, 6);
    graph_t *lstm = LSTM(inputs, 4, SIGMOID, 5);
    writeGraph(lstm);

    lstm = LSTM(inputs, 2, SIGMOID, 5);
    writeGraph(compile(lstm, "lstmDiff"));
    free(inputs);*/

    trainXOR();
    //trainMNISTSimple();
    //trainMNIST();
    return EXIT_SUCCESS;
}