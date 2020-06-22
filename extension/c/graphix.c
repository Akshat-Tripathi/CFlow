#include "../graphix.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "../util.h"

//PRE: length of the current string, and a pointer to the start of the string; node is the node to be drawn
//POST: Correct length of all drawn nodes, and *text of the form a->b;b->c; etc
void drawNode(const node_t *node, int *length, char **text) {
    //Nodes are encoded in this format: NAME[shape={box|circle}];NAME->OUTPUT1,OUTPUT2,...;
    //Calculate new length
    //Alloc space for the NAME
    const int nameLength = strlen(node->name);
    int newLength = nameLength;
    
    //Alloc space for the [shape={box|circle}];
    newLength += 9 + ((node->isData) ? 3 : 6);

    //Alloc space for each NAME->...; and remove the trailing comma
    if (node->m > 0) newLength += nameLength + 2;

    for (int i = 0; i < node->m; i++) newLength += strlen(node->outputs[i]->name) + 1;


    if (!(*text = realloc(*text, newLength + *length + 1))) {
        perror("Couldn't draw node");
        exit(EXIT_FAILURE);
    }
    //Add shape info
    strcat(*text, node->name);
    strcat(*text, "[shape=");
    if (node->isData) {
        strcat(*text, "box];");
    } else {
        strcat(*text, "circle];");
    }
    if (node->m > 0) {
        strcat(*text, node->name);
        strcat(*text, "->");
        for (int i = 0; i < node->m - 1; i++) {
            strcat(*text, node->outputs[i]->name);
            strcat(*text, ",");
        }
        strcat(*text, node->outputs[node->m - 1]->name);
        strcat(*text, ";");
    }
    *length += newLength;
}

char *drawGraph(graph_t *graph) {
    char *string = calloc(14, sizeof(char));
    strcat(string, "digraph test{");
    int length = strlen(string);

    node_t **drawn = NULL;
    int nDrawn = 0;
    node_t *node;
    
    node_t **stack = malloc(sizeof(node_t*) * graph->n);
    if (!memcpy(stack, graph->entryPoints, sizeof(node_t*) * graph->n)) {
        perror("copy error");
        exit(EXIT_FAILURE);
    }
    int stackSize = graph->n;

    while (stackSize) {
        node = pop(&stack, &stackSize);
        if (!contains(drawn, nDrawn, node)) {
            drawNode(node, &length, &string);
            push(&drawn, &nDrawn, node);
            for (int i = 0; i < node->m; i++) {
                push(&stack, &stackSize, node->outputs[i]);
            }
        }
    }	

    free(drawn);
    string = realloc(string, length + 2);
    string[length + 1] = 0;
    string[length] = '}';
    return string;
}

void writeGraph(graph_t *graph) {
    FILE *file = fopen(graph->name, "w");
    fprintf(file, "%s\n", drawGraph(graph));
    fclose(file);
}