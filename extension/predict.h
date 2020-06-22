#ifndef _predict_h_
#define _predict_h_

#include <stdarg.h>

#include "nodes.h"

enum executionMode {
    FORWARD,
    BACKWARD,
    UPDATE
};

void execute(node_t **nodes, int length, enum executionMode mode, 
    void (*optimiser)(node_t *weight, int nArgs, ...), int nArgs, ...);

#endif
