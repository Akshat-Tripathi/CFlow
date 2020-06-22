#ifndef _readCSV_h_
#define _readCSV_h_

#include "matrix.h"

typedef struct csvDataPack {
    double* labels; //array of labels
    matrix2d_t** matrixInputs; // array of matrix pointers
} csvDataPack_t;

csvDataPack_t readCSV(char* filename, int nFields);

#endif