#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "../nodes.h"
#include "../util.h"

#define MAX_FLOAT 1

//PRE: A list of strings, with the length and a node pointer as the element
//POST: A boolean returning whether or not the node is in the list
bool contains(node_t **list, int length, node_t *element) {
    if (!list) return false;
    for (int i = 0; i < length; i++) {
        if (element == list[i]) return true;
    }
    return false;
}

//TODO: Make append and push generic

//PRE: A pointer to a list of strings, and its length and the string to be added
//POST: The list now has the element added to the end
void append(char ***list, int *length, char *element) {
    if (*length) {
        *list = realloc(*list, sizeof(char*) * (*length) + 1);
    } else {
        *list = malloc(sizeof(char*));
    }
    if (!*list) {
        perror("append error");
        exit(EXIT_FAILURE);
    }
    (*list)[(*length)++] = element;
}

//PRE: A pointer to the stack and the length of the stack and the element to be added
//POST: The stack would have the element pushed
void push(node_t ***stack, int *length, node_t *element) {
    if (*length) {
        *stack = realloc(*stack, sizeof(node_t*) * ((*length) + 1));
    } else {
        *stack = malloc(sizeof(node_t*));
    }
    if (!*stack) {
        perror("push error");
        exit(EXIT_FAILURE);
    }
    (*stack)[(*length)++] = element;
}

//PRE: A pointer to the stack and the length of the stack
//POST: The element at the top of the stack would have been popped
node_t *pop(node_t ***stack, int *length) {
    node_t *element = (*stack)[*length - 1];
    if (--(*length)) {
        *stack = realloc(*stack, sizeof(node_t*) *(*length));
    } else {
        *stack = NULL;
    }
    return element;
}

node_t *popFirst(node_t ***stack, int *length) {
    assert(*length >= 0);
    node_t *element = (*stack)[0];
    (*length)--;
    if (!(*length)) {
        free(*stack);
        (*stack) = NULL;
    } else {
        memmove(*stack, (*stack) + 1, (*length) * sizeof(node_t*));
    }
    return element;
}

void freeStack(node_t ***stack, int length) {
	for (int i = 0; i < length; i++) {
		free((*stack)[0]);
	}
	free(stack);
}


//PRE: A valid matrixFunction
//POST: The appropriate encoding of the function as a string
char *encodeOperation(enum matrixFunction func) {
    switch (func) {
        case ADD:             return "ADD";
		case ACTIVATION:	  return "ACTIVATION";
        case AVERAGE_POOLING: return "AVERAGE_POOLING";
        case MULTIPLY:        return "MULTIPLY";
        case MAX_POOLING:     return "MAX_POOLING"; 
        case DOT:             return "DOT"; 
        case DECONVOLUTION:   return "DECONVOLUTION";
		case DILATE: 		  return "DILATE";
		case ROTATE: 		  return "ROTATE";
        case CONVOLUTION:     return "CONVOLUTION"; 
        case SUBTRACT:        return "SUBTRACT";
        case TRANSPOSE: 	  return "TRANSPOSE";
    }
    return "INVALID OPERATION FUNCTION";
}

//PRE: A valid encoding of a matrixFunction
//POST: The appropriate matrixFunction
enum matrixFunction decodeOperation(char *string) {
    switch (*string) {
		case 'R': return ROTATE;
        case 'A':
            if ('D' == *(++string)) return ADD;
			if ('C' == *(string)) return ACTIVATION;
            return AVERAGE_POOLING;
        case 'C': return CONVOLUTION;
        case 'D':
            if ('O' == *(++string)) return DOT;
			if ('I' == *(++string)) return DILATE;
            return DECONVOLUTION;
        case 'M': 
            if ('U' == *(++string)) return MULTIPLY;
            return MAX_POOLING;
        case 'T':	return TRANSPOSE;
        case 'S': 
			return SUBTRACT;	
    }
    return 0;
}

//TODO: Convert to He initialisation
//POST: Generates random float between [-MAX_FLOAT/2..MAX_FLOAT/2]
double randFloat() {
    return ((float)rand() / (float)(RAND_MAX / MAX_FLOAT)) - (MAX_FLOAT / 2);
}

void matrixRandomise(matrix2d_t *matrix) {
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            matrixSet(matrix, i, j, randFloat());
        }
    }
}

//POST: Generates random float between [-MAX_FLOAT/2..MAX_FLOAT/2]
void matrixRandomisePositive(matrix2d_t *matrix) {
    for (int i = 0; i < matrix->nRows; i++) {
        for (int j = 0; j < matrix->nCols; j++) {
            matrixSet(matrix, i, j, fabs(randFloat()));
        }
    }
}