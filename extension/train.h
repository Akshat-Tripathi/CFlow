#ifndef _train_h_
#define _train_h_

#include "matrix.h"
#include "optimisers.h"
#include "error.h"
#include "nodes.h"

void train(graph_t *graph, matrix2d_t **inputs, matrix2d_t **targets,
           double lRate, int epochs, enum errorFunction func,
           int batchSize, enum optimiser optimiser);

#endif