#ifndef _testUtils_h_
#define _testUtils_h_

#include "nodes.h"

char *genName(void);
data_t *initData(bool internalNode, matrix2d_t* matrix);
void link(node_t *node, node_t **inputs, node_t **outputs);
graph_t *genGraph(void);
graph_t *genAdd(void);
graph_t *genDoubleAdd(void);
graph_t *genDot(void);
graph_t *genDense(void);
bool graphsAreEqual(graph_t *graph1, graph_t *graph2);
bool dataAreEqual(data_t *data1, data_t *data2);
matrix2d_t *matrixClone(matrix2d_t *matrix);
void printMatrix(matrix2d_t *matrix);

#endif
