#include "../nodes.h"

#include <stdlib.h>

node_t *nodeInit(char *name, int numInputs, int numOutputs, bool isData) {
    node_t *newNode = malloc(sizeof(node_t));
    newNode->name = name;
    newNode->isData = isData;
    newNode->inputs = calloc(numInputs, sizeof(node_t *));
    newNode->outputs = calloc(numOutputs, sizeof(node_t *));
    newNode->inputIdx = 0;
    newNode->outputIdx = 0;
    newNode->n = numInputs;
    newNode->m = numOutputs;
    newNode->matrix = malloc(sizeof(matrix_t));
    newNode->optimiserMatrix = NULL;

    if (isData) {
        newNode->content.data = malloc(sizeof(data_t));
        newNode->content.data->internalNode = true;
        newNode->content.data->data = malloc(sizeof(matrix_t));
    }
    return newNode;
}

void freeNode(node_t *root) {
    if (!root) return;
    if (!root->name) return;

    for (int i = 0; i < root->m; i++) {
        freeNode(root->outputs[i]);
    }

    if (root->n) free(root->inputs);
    if (root->m) free(root->outputs);
    if (root->matrix->matrix2d)
        matrixFree(root->matrix->matrix2d);
    if (root->optimiserMatrix->matrix2d)
        matrixFree(root->optimiserMatrix->matrix2d);
    free(root);
}

void freeGraph(graph_t *graph) {
    if (!graph) return;

    for (int i = 0; i < graph->n; i++) {
        freeNode(graph->entryPoints[i]);
    }

    free(graph->entryPoints);
    free(graph->exitPoints);
    free(graph);
}

void linkNodes(node_t *input, node_t *output) {
    input->outputs[(input->outputIdx)++] = output;
    output->inputs[(output->inputIdx)++] = input;
}

graph_t *graphInit(char *name, int n, node_t **entryPoints, 
                               int m, node_t **exitPoints) {
    graph_t *tmp = malloc(sizeof(graph_t));
    tmp->name = name;
    tmp->n = n;
    tmp->m = m;
    tmp->entryPoints = entryPoints;
    tmp->exitPoints = exitPoints;
    return tmp;
}

// Tests
/*
int main() {
    node_t *n1 = nodeInit("ASD", 0, 1);
    node_t *n2 = nodeInit("B", 1, 0);
    n1->outputs[0] = n2;
    n2->inputs[0] = n1;
    freeNode(n1);
    freeNode(n2);
}
*/