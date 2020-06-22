#ifndef _nodes_h_
#define _nodes_h_

#include <stdbool.h>

#include "matrix.h"
#include "activation.h"

#define MAX_NODE_NAME_LENGTH 15

typedef struct operation {
    enum matrixFunction funcName;
    enum activationFunction activationName;
} operation_t;

typedef union {
	matrix2d_t *matrix2d;
	matrix3d_t *matrix3d;
} matrix_t;

typedef struct data {
    bool internalNode;
    matrix_t *data;
} data_t;

typedef union {
    data_t *data;
    operation_t operation;
} nodeType_t; 

typedef struct node {
    char *name;
    bool isData; //Used to decide the type in content
    nodeType_t content;
    int n, m;
    struct node **inputs;
    struct node **outputs;
    int inputIdx, outputIdx;
    matrix_t *matrix;
    matrix_t *optimiserMatrix; //Used for storing velocities/gradient accumalations
    matrix_t *poolingMatrixGrad;
} node_t;

//PRE: All graphs are acyclic
typedef struct graph {
    char *name;
    node_t **entryPoints;
    int n, m;
    //This is to help align targets with their appropriate nodes in train
    node_t **exitPoints;
} graph_t;

node_t *nodeInit(char* name, int numInputs, int numOutputs, bool isData);
void linkNodes(node_t *input, node_t *output);
void freeNode();
graph_t *graphInit(char* name, int n, node_t **entryPoints, 
                               int m, node_t **exitPoints);
void freeGraph(graph_t *graph);

#endif