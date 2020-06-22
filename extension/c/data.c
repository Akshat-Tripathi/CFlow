#include "../data.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define INITIAL_N_DATA_BLOCKS 10

//PRE: file is the file to be appended to (MUST be opened in append mode), node is a node_t which isn't an operation 
void writeBlock(FILE *file, node_t node) {
    data_t data = *(node.content.data);
    //Write name to file
    fwrite(node.name, sizeof(char), strlen(node.name), file);

	//Write length, width to the file
	fprintf(file, "\t%d\t%d\t", data.data->matrix2d->nRows, data.data->matrix2d->nCols);

    //Write matrix values to the file
	for (int i = 0; i < data.data->matrix2d->nRows; i++)
        fwrite(data.data->matrix2d->data[i], sizeof(double), data.data->matrix2d->nCols, file);
}

//PRE: file must be in rb mode, string must be large enough to accomodate the data
static char *seek(FILE *file) {
    char *string = calloc(MAX_NODE_NAME_LENGTH + 1, sizeof(char));
    char c;
    int length = 0;
    int currentMax = MAX_NODE_NAME_LENGTH;
    do {
        c = fgetc(file);
        if (length >= currentMax) {
            currentMax <<= 2;
            string = realloc(string, currentMax * sizeof(char));
        }
        string[length++] = c;
    } while (c != '\t');

    string[length - 1] = 0;
    string = realloc(string, length * sizeof(char));
    return string;
}

//PRE: File is in read mode and is in the format which writeBlock produces
//POST: Returns the data in the block, unserialised into a struct
static data_t *readBlock(FILE *file) {
    // char *name = malloc(MAX_NODE_NAME_LENGTH + 1);
    // char *cRows = malloc(4); //TODO standardise this
    // char *cCols = malloc(4); //TODO standardise this
    // seek(file, &name);
    // seek(file, &cRows);
    // seek(file, &cCols);

    seek(file);
    char *cRows = seek(file);
    char *cCols = seek(file);
    int rows = atoi(cRows);
    int cols = atoi(cCols);
    matrix2d_t *matrix = matrixCreate(rows, cols);
    for (int i = 0; i < rows; i++) {
        fread(matrix->data[i], sizeof(double), cols, file);
    }

    data_t *data = malloc(sizeof(data_t));
    data->data = malloc(sizeof(matrix_t));
    data->internalNode = true;
    data->data->matrix2d = matrix;
    return data;
}

//PRE: Takes in a file produced by writeBlocks
//POST: Returns all data blocks in the file
data_t **readData(char *filename) {
    FILE *file = fopen(filename, "rb");
    int length = 0, currentLength = INITIAL_N_DATA_BLOCKS;
    data_t **blocks = calloc(currentLength, sizeof(data_t*));

    char c = getc(file);
    while (!feof(file)) {
        ungetc(c, file);
        if (length >= currentLength) {
            //If the max length has been hit, double it
            currentLength *= 2;
            blocks = realloc(blocks, sizeof(data_t*) * currentLength);
        }
        blocks[length++] = readBlock(file);
        c = getc(file);
    }
    blocks = realloc(blocks, sizeof(data_t*) * length);
    fclose(file);
    return blocks;
}