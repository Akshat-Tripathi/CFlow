#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../readCSV.h"
#include "../matrix.h"

double* doubleTokenize(char* str, int nTokens) {
    double* tokens = calloc(nTokens, sizeof(double));
    int i = 0;
    str = strtok(str, ",");

    while (str != NULL) {
        tokens[i++] = (double) atoi(str);
        str = strtok(NULL, ",");
    }

    return tokens;
}

#define SHAPE_SIZE 28
#define MAX_LINE_SIZE 10000

csvDataPack_t readCSV(char* filename, int nFields) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open file\n");
        exit(EXIT_FAILURE);
    }
    matrix2d_t** matrixRecord = calloc(nFields, sizeof(matrix2d_t*));
    double* labelRecord = calloc(nFields, sizeof(double));


    char buffer[MAX_LINE_SIZE];
    fgets(buffer, MAX_LINE_SIZE, fp);
    for (int i = 0; i < nFields; i++) {
        double* tokens = doubleTokenize(strtok(fgets(buffer, MAX_LINE_SIZE, fp),  "\n"), (SHAPE_SIZE*SHAPE_SIZE) + 1);
        labelRecord[i] = tokens[0];
        matrix2d_t *tmp = matrixCreate(SHAPE_SIZE, SHAPE_SIZE);
        for (int i = 0; i < SHAPE_SIZE; i++) {
            for (int j = 0; j < SHAPE_SIZE; j++) {
                matrixSet(tmp, i, j, tokens[1 + (i * SHAPE_SIZE) + j]);
            }
        }
        matrixRecord[i] = tmp;
    }

    csvDataPack_t returnValue = {.labels = labelRecord, .matrixInputs = matrixRecord};
    fclose(fp);
    return returnValue;
}