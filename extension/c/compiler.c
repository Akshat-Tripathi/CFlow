#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

#include "../testUtils.h"
#include "../util.h"
#include "../nodes.h"
#include "../graphix.h"
#include "../activation.h"
#include "../layers.h"

#define ASCII_ZERO 48

node_t *_differentiate(node_t *node, node_t ***entryPoints, int *nLoss);

int nDot = 0;
int nTranspose = 0;
int nAdd = 0;
int nConv = 0;
int nMultiply = 0;

static char *encode(enum matrixFunction funcName) {
    char *str = encodeOperation(funcName);
    char *numStr = malloc(strlen(str) + 3); //Assuming n* doesn't exceed 2 digits
    switch (funcName) {
        case DOT: 		  sprintf(numStr, "%s%d", str, nDot++); break;
        case ADD: 		  sprintf(numStr, "%s%d", str, nAdd++); break;
        case MULTIPLY:    sprintf(numStr, "%s%d", str, nMultiply++); break;
        case TRANSPOSE:   sprintf(numStr, "%s%d", str, nTranspose++); break;
        case CONVOLUTION: sprintf(numStr, "%s%d", str, nConv++); break;
        default:          return NULL;
    }
    return numStr;
}

static void linkDeriv(node_t *root, node_t *leaf) {
    if (strcmp("dummy", leaf->name)) {
        root->outputs[root->outputIdx++] = leaf;
        leaf->inputs[leaf->inputIdx++] = root;
    } else {
        //Move outputs of the leaf to the root
        root->m += leaf->m - 1;
        root->outputs = realloc(root->outputs, root->m * sizeof(node_t*));
        node_t *next;
        for (int i = 0; i < leaf->m; i++) {
            next = leaf->outputs[i];
            root->outputs[root->outputIdx++] = next;
            //Replace everything with an input to this dummy
            for (int j = 0; j < next->n; j++) {
                if (next->inputs[j] == leaf) {
                    next->inputs[j] = root;
                }
            }
            
        }
        free(leaf);
    }
}

node_t *clone(node_t *node) {
    node_t *input = nodeInit(node->name, node->n, 1, node->isData);
    input->content = node->content;
    for (int i = 0; i < node->n; i++) {
        input->inputs[i] = node->inputs[i];
    }
    input->matrix = node->matrix;
    return input;
}

node_t *_productRule(node_t *deltaNode, node_t *forwardNode, enum matrixFunction funcName, node_t ***lossPoints, int *nLoss) {
    push(lossPoints, nLoss, deltaNode);
    node_t *op = nodeInit(encode(funcName), funcName == CONVOLUTION ? 3 : 2, 1, false);
    op->content.operation.funcName = funcName;
    forwardNode->outputs[0] = op;
    linkDeriv(op, _differentiate(deltaNode, lossPoints, nLoss));

    op->inputs[op->inputIdx++] = forwardNode;
    return op;
}

node_t *productRule(node_t *first, node_t *second, enum matrixFunction funcName, node_t ***lossPoints, int *nLoss) {
    node_t *derivative = nodeInit("dummy", 1, 2, false);
    if (DOT == funcName) {
        linkDeriv(derivative, _productRule(first->inputs[0], second, funcName, lossPoints, nLoss));
        linkDeriv(derivative, _productRule(second->inputs[0], first, funcName, lossPoints, nLoss));	
    } else {
        linkDeriv(derivative, _productRule(first, second, funcName, lossPoints, nLoss));
        linkDeriv(derivative, _productRule(second, first, funcName, lossPoints, nLoss));
    }
    return derivative;
}

//TODO: Refactor into a new productRule func
//PRE: Nodes form a tree
//PRE: the first node is the node before the end result of a subgraph
//POST: A tree where representing the derivative of the input
node_t *_differentiate(node_t *node, node_t ***lossPoints, int *nLoss) {
    node_t *derivative;
    if (!strcmp(node->name, "noop")) {
        return _differentiate(node->inputs[0], lossPoints, nLoss);
    } else if (node->isData) {
        //Needs to have the same underlying data
        char *newName = malloc(strlen(node->name) + 2);
        newName[0] = 'd';
        newName[1] = 0;
        derivative = nodeInit(strcat(newName, node->name), node->m, 0, true);
        derivative->content = node->content;
    } else {
        char *name;
        switch (node->content.operation.funcName) {
            case ADD:
                //Init dummy node with 2 outputs and 1 input
                //PRE: ADD operations only have 2 inputs
                derivative = nodeInit("dummy", 1, 2, false);
                linkDeriv(derivative,  _differentiate(node->inputs[0], lossPoints, nLoss));
                linkDeriv(derivative,  _differentiate(node->inputs[1], lossPoints, nLoss));
                break;
            case MULTIPLY:
                //Product rule A x B
                derivative = productRule(clone(node->inputs[0]), clone(node->inputs[1]), MULTIPLY, lossPoints, nLoss);
                break;
            case DOT:
                //A dot B
                {node_t *a = clone(node->inputs[0]);
                node_t *transpose1 = nodeInit(encode(TRANSPOSE), 1, 1, false);
                transpose1->content.operation.funcName = TRANSPOSE;
                a->outputs[a->outputIdx++] = transpose1;
                transpose1->inputs[0] = a;

                node_t *b = clone(node->inputs[1]);
                node_t *transpose2 = nodeInit(encode(TRANSPOSE), 1, 1, false);
                transpose2->content.operation.funcName = TRANSPOSE;
                b->outputs[b->outputIdx++] = transpose2;
                transpose2->inputs[0] = b;
                derivative = productRule(transpose2, transpose1, DOT, lossPoints, nLoss);}
                break;
            case CONVOLUTION:
                //This is probably the source of all bugs
                //A convolved with kernel K
                {node_t *k = clone(node->inputs[1]);
                node_t *rotate = nodeInit("rotate", 1, 1, false);
                k->outputs[0] = rotate;
                rotate->inputs[0] = k;

                rotate->content.operation.funcName = ROTATE;

                derivative = productRule(clone(node->inputs[0]), rotate, CONVOLUTION, lossPoints, nLoss);

                //Dilate the derivative when finding dA
                node_t conv = *derivative->outputs[1];
                conv.inputs[2] = nodeInit("config", 0, 1, true);
                //dPadding = Padding - 1; dStride = Stride
                matrix2d_t *args = conv.inputs[2]->matrix->matrix2d;
                matrixSet(args, 0, 1, matrixGet(node->inputs[2]->matrix->matrix2d, 0, 1) - 1);
                matrixSet(args, 0, 0, 1);

                //0th input is the matrix to be dilated, args are in the 1st input
                node_t *dilate = nodeInit("dilate", 2, 1, false);
                dilate->content.operation.funcName = DILATE;

                dilate->outputs[0] = &conv;
                matrixSet(dilate->inputs[1]->matrix->matrix2d, 0, 0, matrixGet(node->inputs[2]->matrix->matrix2d, 0, 0) - 1);

                derivative->outputs[1] = dilate;
                matrix2d_t *mat = derivative->outputs[0]->inputs[2]->matrix->matrix2d;
                
                //1 stride 0 padding
                matrixSet(mat, 0, 0, 1);
                matrixSet(mat, 0, 1, 0);}
                break;
            case MAX_POOLING:
                derivative = nodeInit("dMAXPOOLING", 1, 1, false);
                derivative->content.operation.funcName = MAX_POOLING;
                linkDeriv(derivative, _differentiate(node->inputs[0], lossPoints, nLoss));
                break;
            case AVERAGE_POOLING:
                break;
            case FLATTEN:
                derivative = nodeInit("FLATTEN", 2, 1, false);
                derivative->content.operation.funcName = FLATTEN;

                node_t *config = nodeInit("config", 0, 1, true);
                config->matrix->matrix2d = matrixCreate(1, 3);
                
                matrix3d_t *inputMatrix = node->inputs[0]->matrix->matrix3d;
                matrixSet(config->matrix->matrix2d, 0, 0, inputMatrix->nRows);
                matrixSet(config->matrix->matrix2d, 0, 1, inputMatrix->nCols);
                matrixSet(config->matrix->matrix2d, 0, 2, inputMatrix->nDepth);
                derivative->inputs[1] = config;

                linkDeriv(derivative,  _differentiate(node->inputs[0], lossPoints, nLoss));
                break;
            default:
            //TODO: efficientise
                name = calloc(strlen(node->name) + 2, sizeof(char));
                name[0] = 'd';
                name = strcat(name, node->name);
                derivative = nodeInit(name, 1, 1, false);
                derivative->content.operation.funcName = node->content.operation.funcName;
                derivative->content.operation.activationName = getDeriv(node->content.operation.activationName);
                linkDeriv(derivative,  _differentiate(node->inputs[0], lossPoints, nLoss));
        }
    }
    return derivative;
}

//PRE: EndPoint must have 1 input and must be a tree
//POST: Returns the derivative of the input
node_t *differentiate(node_t *endPoint, node_t ***lossPoints, int *nLoss) {
    node_t *deriv = clone(endPoint);
    deriv->outputs = malloc(sizeof(node_t*));
    deriv->m = 1;
    linkDeriv(deriv, _differentiate(endPoint->inputs[0], lossPoints, nLoss));
    return deriv;
}

node_t *insertNoOp(node_t *node, int outputIdx) {
    node_t *other = node->outputs[outputIdx];

    node_t *noOp = nodeInit("noop", 1, 1, false);
    noOp->inputs[0] = node;
    noOp->outputs[0] = other;

    for (int i = 0; i < other->n; i++) {
        if (other->inputs[i] == node) {
            other->inputs[i] = noOp;
            break;
        }
    }

    return noOp;
}

//Compilation algorithm
//1. DFS through the graph, differentiate when a node with 0 outputs is reached
//2. If the node has more than 1 output:
//		Add noops between the current node, and each of its outputs
//		Compile from the noops until the end
//		Remove the noops and link the nodes back into the main graph	

//PRE: node doesn't have 1 output
//POST:
void _compile(node_t *node, node_t ***compiled, int *nCompiled, node_t ***lossPoints, int *nLoss) {
    node_t *diff;
    if (!contains(*compiled, *nCompiled, node)) {
        if (strcmp("noop", node->name)) push(compiled, nCompiled, node);
        if (1 == node->m) {
            _compile(node->outputs[0], compiled, nCompiled, lossPoints, nLoss);
        } else if (!node->m) {
            diff = differentiate(node, lossPoints, nLoss);
            push(lossPoints, nLoss, diff);
        } else {
            //1. Insert a noOp for every output of the node
            //Need 2 for loops in case you have x + x

            node_t *copy = clone(node);
            copy->m = 0;
            node_t **noOps = malloc(sizeof(node_t*) * node->m);
            for (int i = 0; i < node->m; i++) noOps[i] = insertNoOp(node, i);
            
            //2. Differentiate input node
            diff = _differentiate(node, lossPoints, nLoss);

            //3. For each noOp compile the graph from it
            node_t **subGraphLossPoints;
            int subGraphNLoss;
            node_t *point;

            for (int i = 0; i < node->m; i++) {
                subGraphLossPoints = NULL;
                subGraphNLoss = 0;
                _compile(noOps[i], compiled, nCompiled, &subGraphLossPoints, &subGraphNLoss);
                for (int j = 0; j < subGraphNLoss; j++) {
                    point = subGraphLossPoints[j];
                    //Remove noOp
                    if (!strcmp("noop", point->name)) {
                        for (int k = 0; k < point->m; k++) {
                            push(&(copy->outputs), &(copy->m), point->outputs[k]);
                            for (int l = 0; l < point->outputs[k]->n; l++) {
                                if (point->outputs[k]->inputs[l] == point) {
                                    point->outputs[k]->inputs[l] = copy;
                                }
                            }
                        }
                        push(lossPoints, nLoss, copy);
                    } else {
                        push(lossPoints, nLoss, point);
                    }
                }
            }
        }
    }
}

//PRE: Takes in a pointer to a graph - forward pass
//POST: Returns the computational graph to be used during backpropagation
graph_t *compile(graph_t *graph, char *name) {
    node_t **compiled = NULL;
    int nCompiled = 0;

    node_t **lossPoints = NULL;
    int nLoss = 0;

    for (int i = 0; i < 1; i++) {
        _compile(graph->entryPoints[i], &compiled, &nCompiled, &lossPoints, &nLoss);
    }

    return graphInit(name, nLoss, lossPoints, 0, NULL);
}

/*
int main(void) {
    node_t **entryPoints = malloc(sizeof(node_t*) * 2);
    entryPoints[0] = nodeInit("x0", 0, 4, true);
    entryPoints[0]->matrix = matrixCreate(1, 2);
    entryPoints[1] = nodeInit("x1", 0, 4, true);
    entryPoints[1]->matrix = matrixCreate(1, 2);
    entryPoints[2] = nodeInit("x2", 0, 4, true);
    entryPoints[2]->matrix = matrixCreate(1, 2);

    graph_t *graph = LSTM(entryPoints, 3, SIGMOID, 3);
    writeGraph(graph);
    printf("LSTM written");
    graph_t *compiled = compile(graph);
    writeGraph(compiled);
    return EXIT_SUCCESS;
}
*/
/*
int main(void) {
    node_t **entryPoints = NULL;
    int length = 0;

    graph_t *graph = LS;
    
    writeGraph(graph);
    
    graph_t *compiled = compile(graph);
    writeGraph(compiled);

    return EXIT_SUCCESS;
}*/
/*
int main(void) {
    node_t *x = nodeInit("x", 0, 2, true);
    x->matrix = &(matrix2d_t) {.nRows = 3, .nCols = 2};

    node_t *y = nodeInit("y", 1, 0, true);

    node_t *prevOutput = nodeInit("prevOutput", 0, 2, true);
    prevOutput->matrix = &(matrix2d_t) {.nRows = 3, .nCols = 4};


    node_t **entryPoints = malloc(sizeof(node_t*) * 2);
    entryPoints[0] = x;
    entryPoints[1] = prevOutput;
    int length = 2;

    node_t *output = genDouble(x, prevOutput, SIGMOID, &entryPoints, &length);

    output->outputs[0] = y;
    y->inputs[0] = output;

    graph_t graph = (graph_t) {.n = length, .entryPoints = entryPoints, .name = "lstm"};
    
    writeGraph(&graph);
    
    graph_t *compiled = compile(&graph);
    writeGraph(compiled);

    return EXIT_SUCCESS;
}*/

graph_t *lstmEnd(void) {
    node_t *activationGate = nodeInit("activationGate", 0, 1, true);
    node_t *inputGate = nodeInit("inputGate", 0, 1, true);
    node_t *forgetGate = nodeInit("forgetGate", 0, 1, true);
    node_t *outputGate = nodeInit("outputGate", 0, 1, true);
    
    node_t *prevState = nodeInit("prevState", 0, 1, true);

    char *mult1Name = "mult1";
    char *mult2Name = "mult2";
    char *stateName = "state";

    char *tanhName = "tanh";
    char *outputName = "output";

    node_t *mult1 = nodeInit(mult1Name, 2, 1, false);
    mult1->content.operation = (operation_t) {.funcName = MULTIPLY};

    node_t *mult2 = nodeInit(mult2Name, 2, 1, false);
    mult2->content.operation = (operation_t) {.funcName = MULTIPLY};

    node_t *state = nodeInit(stateName, 2, 2, false);
    state->content.operation = (operation_t) {.funcName = ADD};

    node_t *state2 = nodeInit("nextState", 1, 0, true);

    node_t *tanh = nodeInit(tanhName, 2, 1, false);
    tanh->content.operation = (operation_t) {.funcName = ACTIVATION, .activationName = TANH};

    node_t *output = nodeInit(outputName, 2, 0, false);
    output->content.operation = (operation_t) {.funcName = MULTIPLY};
    
    linkNodes(activationGate, mult1);
    linkNodes(inputGate, mult1);

    linkNodes(prevState, mult2);
    linkNodes(forgetGate, mult2);

    linkNodes(mult1, state);
    linkNodes(mult2, state);

    linkNodes(state, tanh);
    linkNodes(state, state2);
    linkNodes(tanh, output);
    linkNodes(outputGate, output);

    node_t **entryNodes = malloc(sizeof(node_t*) * 4);
    entryNodes[0] = activationGate;
    entryNodes[1] = inputGate;
    entryNodes[2] = forgetGate;
    entryNodes[3] = outputGate;

    return graphInit("end", 4, entryNodes, 0, NULL);
}