#ifndef _optimisers_h_
#define _optimisers_h_

#include "nodes.h"
#include "matrix.h"

enum optimiser {
    SGD,
    MOMENTUM,
    ADAGRAD,
    RMSPROP
};

void sgd(node_t *weight, int nArgs, ...);
void sgdMomentum(node_t *weight, int nArgs, ...);
void sgdNesterovMomentum(node_t *weight, int nArgs, ...);
void adagrad(node_t *weight, int nArgs, ...);
void RMSProp(node_t *weight, int nArgs, ...);

#endif