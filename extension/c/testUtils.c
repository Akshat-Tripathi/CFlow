#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../nodes.h"
#include "../scheduler.h"
#include "../util.h"
#include "../testUtils.h"

char a = 65;

char *genName(void) {
    char *str = malloc(sizeof(char) * 2);
    str[0] = a++;
    str[1] = 0;
    return str;
}

data_t *initData(bool internalNode, matrix2d_t *matrix) {
    data_t *data = malloc(sizeof(data_t));
	data->data = malloc(sizeof(matrix_t));
    data->internalNode = internalNode;
    data->data->matrix2d = matrix;
    return data;
}

//AVOID using this. Use LinkNodes instead
void link(node_t *node, node_t **inputs, node_t **outputs) {
    for (int i = 0; i < node->n; i++) {
        node->inputs[i] = inputs[i];
    }

    for (int i = 0; i < node->m; i++) {
        node->outputs[i] = outputs[i];
    }
}

graph_t *genGraph(void) {
    matrix2d_t* matrix = matrixCreate(30, 30);
    matrixRandomise(matrix);

    node_t *a = nodeInit("A", 0, 1, true);
    a->content.data->data->matrix2d = matrix;
    node_t *b = nodeInit("B", 0, 1, true);
    b->content.data->data->matrix2d = matrix;
    node_t *c = nodeInit("C", 0, 2, true);
    c->content.data->data->matrix2d = matrix;
    node_t *d = nodeInit("D", 2, 2, true);
    d->content.data->data->matrix2d = matrix;
    node_t *e = nodeInit("E", 1, 0, true);
    e->content.data->data->matrix2d = matrix;
    node_t *f = nodeInit("F", 2, 0, true);
    f->content.data->data->matrix2d = matrix;
    node_t *g = nodeInit("G", 1, 0, true);
    g->content.data->data->matrix2d = matrix;

    linkNodes(a, d);
    linkNodes(b, d);
    linkNodes(c, f);
    linkNodes(c, g);
    linkNodes(d, e);
    linkNodes(d, f);

    graph_t *graph = malloc(sizeof(graph_t));
    graph->n = 3;
    graph->entryPoints = malloc(sizeof(node_t *) * 3);
    graph->entryPoints[0] = a;
    graph->entryPoints[1] = b;
    graph->entryPoints[2] = c;
    graph->exitPoints = calloc(3, sizeof(node_t*));
    graph->exitPoints[0] = e;
    graph->exitPoints[1] = f;
    graph->exitPoints[2] = g;
    graph->name = "Test";

    return graph;
}

bool nodesAreEqual(node_t *node1, node_t *node2) {
    bool eq = strcmp(node1->name, node2->name) == 0 &&
              (node1->isData == node2->isData) &&
              (node1->n == node2->n) &&
              (node1->m == node2->m);
    if (!eq) return false;
    for (int i = 0; i < node1->n; i++)
        if (strcmp(node1->inputs[i]->name, node2->inputs[i]->name) != 0) {
            return false;
        }

    for (int i = 0; i < node1->m; i++)
        if (strcmp(node1->outputs[i]->name, node2->outputs[i]->name) != 0)
            return false;

    return true;
}

// Checks if two data objects are equal
bool dataAreEqual(data_t *data1, data_t *data2) {
    assert(data1);
    assert(data2);
    if (data1->internalNode == data2->internalNode) {
        if (data1->data->matrix2d->nRows == data2->data->matrix2d->nRows &&
            data1->data->matrix2d->nCols == data2->data->matrix2d->nCols) {
            for (int i = 0; i < data1->data->matrix2d->nRows; i++) {
                for (int j = 0; j < data1->data->matrix2d->nCols; j++) {
                    if (matrixGet((data1->data->matrix2d), i, j) != matrixGet((data2->data->matrix2d), i, j)) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
    return false;
}

graph_t *genDoubleAdd(void) {
    node_t *x = nodeInit("x", 0, 2, true);
    node_t *y = nodeInit("y", 1, 0, true);
    node_t *add = nodeInit("add0", 2, 2, false);
    node_t *add1 = nodeInit("add1", 2, 1, false);

    x->outputs[0] = x->outputs[1] = add;
    x->content.data->data->matrix2d = matrixCreate(3, 4);
    add->inputs[0] = add->inputs[1] = x;

    add->outputs[0] = add->outputs[1] = add1;

    add1->inputs[0] = add1->inputs[1] = add;
    add1->outputs[0] = y;
    y->inputs[0] = add1;

    node_t **entryPoints = NULL;
    int length = 0;
    push(&entryPoints, &length, x);

    return graphInit("doubleAdd", 1, entryPoints, 0, NULL);
}

graph_t *genDense(void) {
	node_t *w = nodeInit("w", 0, 1, true);
	node_t *x = nodeInit("x", 0, 2, true);
	node_t *y = nodeInit("y", 1, 0, true);
	//node_t *z = initNode("z", 1, 0, true);

	node_t *dot = nodeInit("dot", 2, 1, false);
	node_t *add = nodeInit("add", 2, 1, false);
	node_t *sigmoid = nodeInit("sigmoid", 1, 1, false);

	dot->content.operation.funcName = DOT;
	add->content.operation.funcName = ADD;
	sigmoid->content.operation.funcName = ACTIVATION;
	sigmoid->content.operation.activationName = SIGMOID;

	w->outputs[0] = dot;
	x->outputs[0] = dot;
	x->outputs[1] = add;
	
	dot->inputs[0] = x;
	dot->inputs[1] = w;
	dot->outputs[0] = add;

	add->inputs[0] = dot;
	add->inputs[1] = x;

	add->outputs[0] = sigmoid;

	sigmoid->inputs[0] = add;
	sigmoid->outputs[0] = y;
	//sigmoid->outputs[1] = z;

	y->inputs[0] = sigmoid;
	//z->inputs[0] = sigmoid;

	graph_t *layer = malloc(sizeof(graph_t));
	layer->n = 2;
	layer->entryPoints = malloc(sizeof(node_t*) * layer->n);
	layer->entryPoints[0] = x;
	layer->entryPoints[1] = w;
	layer->name = "dense";

	return layer;
}

void printMatrix(matrix2d_t *matrix) {
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            printf("%lf ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

bool graphsAreEqual(graph_t *graph1, graph_t *graph2) {
    int length1, length2;
    node_t **nodes1 = schedule(graph1, &length1);
    node_t **nodes2 = schedule(graph2, &length2);

    if (length1 != length2) return false;
    for (int i = 0; i < length1; i++) {
        if (!nodesAreEqual(nodes1[i], nodes2[i])) return false;
    }
    return true;
}

matrix2d_t *matrixClone(matrix2d_t *matrix) {
	matrix2d_t *result = matrixCreate(matrix->nRows, matrix->nCols);
	for (int i = 0; i < matrix->nRows; i++) {
		for (int j = 0; j < matrix->nCols; j++) {
			matrixSet(result, i, j, matrixGet(matrix, i, j));
		}
	}
	return result;
}
