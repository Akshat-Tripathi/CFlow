#ifndef _data_h_
#define _data_h_

#include <stdbool.h>
#include <stdio.h>

#include "nodes.h"
#include "matrix.h"

void writeBlock(FILE *file, node_t node);
data_t **readData(char *filename);

#endif