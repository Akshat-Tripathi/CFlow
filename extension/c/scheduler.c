#include "../scheduler.h"

#include <stdlib.h>
#include "../nodes.h"
#include "../util.h"
#include "../testUtils.h"

static node_t **schedule_helper(node_t *node, node_t ***exited, int *length, int *nNodes) {
    node_t **nodes = NULL;
    *nNodes = 0;
    int n = 0;
    node_t **newNodes;
    //Attempt to dfs through the outputs of this node which haven't been exited
    for (int i = 0; i < node->m; i++) {
        if (!contains(*exited, *length, node->outputs[i])) {
            //Recurse and add the result to nodes and exited
            newNodes = schedule_helper(node->outputs[i], exited, length, nNodes);
            for (int j = 0; j < *nNodes; j++) {
                push(&nodes, &n, newNodes[j]);
            }
            free(newNodes);
        }
    }
    //Add this node to exited and nodes
    push(exited, length, node);
    push(&nodes, &n, node);
    *nNodes = n;
    return nodes;
}

node_t **schedule(graph_t *graph, int *nNodes) {
    node_t **nodes = NULL, **newNodes;
    int nNewNodes, length = 0;
    *nNodes = 0;
    node_t **exited;

    for (int i = 0; i < graph->n; i++) {
        nNewNodes = 0;
        newNodes = schedule_helper(graph->entryPoints[i], &exited, &length, &nNewNodes);
        for (int j = 0; j < nNewNodes; j++) {
            push(&nodes, nNodes, newNodes[j]);
        }
    }
    
    //free(exited);
    free(newNodes);

    return nodes;
}