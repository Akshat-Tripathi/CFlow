#ifndef _util_h_
#define _util_h_

#include "nodes.h"

bool contains(node_t **list, int length, node_t *element);
void append(char ***list, int *length, char *element);
void push(node_t ***stack, int *length, node_t *element);
node_t *pop(node_t ***stack, int *length);
node_t *popFirst(node_t ***stack, int *length);
void freeStack(node_t ***stack, int length);
char *encodeOperation(enum matrixFunction func);
enum matrixFunction decodeOperation(char *string);
double randFloat();
void matrixRandomise(matrix2d_t *matrix);
void matrixRandomisePositive(matrix2d_t *matrix);
#endif