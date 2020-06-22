#ifndef _file_h_
#define _file_h_ 

#include "nodes.h"

#define NUM_ALPHANUMERIC 62

typedef struct hashtable {
    int val;
    bool isWord;
    struct hashtable *bucket[NUM_ALPHANUMERIC];
} hashtable_t;

void graphFileWrite(char* filename, graph_t* graphToWrite);
graph_t *graphFileRead(char* filename);

#endif