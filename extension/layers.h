#ifndef _layers_h_
#define _layers_h_

#include "nodes.h"

node_t *denseLayer(node_t *x, int nNeurons,
        enum activationFunction activationFunction, 
        node_t ***entryPoints, int *length);

graph_t *LSTM(node_t **inputs, int timeSteps,
              enum activationFunction func, int nNeurons);
#endif