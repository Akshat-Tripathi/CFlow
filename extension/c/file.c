#include "../file.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data.h"
#include "../matrix.h"
#include "../testUtils.h"
#include "../util.h"

// 32-bit integer + null char
#define INT_SIZE 33

//TODO: 512 ARBITRARY CHOICE
#define MAX_LINE_LENGTH 512

void freeHashtable(hashtable_t* table) {
    if (!table) return;

    for (int i = 0; i < NUM_ALPHANUMERIC; i++) {
        freeHashtable(table->bucket[i]);
    }

    free(table);
}

//PRE: A char that you want to hash. Assumes the char is alphanumeric
//POST: Returns the hash of that char as indicated below
// 0-9:0-9; A-Z:10-35; a-z:36-61
int hash(char key) {
    int offset = 10;
    if (48 <= key && key <= 57) {
        return key - 48;
    } else if (65 <= key && key <= 90) {
        return key - 65 + offset;
    } else {
        return key - 97 + 26 + offset;
    }
}

//PRE: A hashtable, a string you want to store and the int value for the string
//POST: A bool indicating whether add was successful or not
bool add(hashtable_t* table, char* key, int value) {
    hashtable_t* currTable = table;

    while (*key != '\0') {
        if (currTable->bucket[hash(*key)] == NULL) {
            currTable->bucket[hash(*key)] = calloc(1, sizeof(hashtable_t));
            if (!currTable->bucket[hash(*key)]) {
                perror("Bucket allocation failed");
                return false;
            }
        }

        currTable = currTable->bucket[hash(*key)];
        key++;
    }

    currTable->val = value;
    currTable->isWord = true;
    return true;
}

//PRE: A hashtable, a string you want to search for, string is stored in hashtable
//POST: The value of the string key
int get(hashtable_t* table, char* key) {
    hashtable_t* currTable = table;

    while (*key != '\0') {
        currTable = currTable->bucket[hash(*key)];
        key++;
    }

    return currTable->val;
}

//PRE: Takes in node and a string representing the filename that data_t is stored in
//POST: Returns a string-encoded node
char* encodeNode(node_t* node, FILE* fp, char*** linkLines, int* linksCreated) {
    //FORMAT: NODE NODE_NAME isData(1/0) DATA_FILENAME/OP_NAME N M
    char* contentName;

    if (node->isData) {
        writeBlock(fp, *node);
        contentName = "DATA";
    } else {
        contentName = encodeOperation(node->content.operation.funcName);
    }

    for (int i = 0; i < node->m; i++) {
        char* linkLine = calloc(MAX_LINE_LENGTH, sizeof(char));
        sprintf(linkLine, "LINK %s %s\n", node->name, node->outputs[i]->name);
        // printf("%s", linkLine);
        append(linkLines, linksCreated, linkLine);
    }

    char* encoded = calloc(MAX_LINE_LENGTH, sizeof(char));
    sprintf(encoded, "%s %s %d %s %d %d", "NODE", node->name, node->isData, contentName, node->n, node->m);
    encoded = realloc(encoded, strlen(encoded) + 1);
    return encoded;
}

char** encodeGraph(graph_t* graph, char*** linkSection, int* nNodeLines, int* nLinks) {
    char** lines = calloc(1, sizeof(char*));
    int nLines = 0;

    //dataFilename: GRAPHNAME_DATA\0
    //Each graph stores the data field of all nodes in 1 file
    char* dataFilename = calloc(strlen(graph->name + 1) + 6, sizeof(char));
    sprintf(dataFilename, "%s_data", graph->name);
    FILE* dataFile = fopen(dataFilename, "a");
    append(&lines, &nLines, dataFilename);

    node_t** visited = NULL;
    int nVisited = 0;

    node_t** stack = malloc(graph->n * sizeof(node_t*));
    int stackSize = graph->n;
    if (!memcpy(stack, graph->entryPoints, sizeof(node_t*) * graph->n)) {
        perror("Memcpy error");
        exit(EXIT_FAILURE);
    }

    char** linkLines = NULL;
    int linksCreated = 0;

    node_t* node;
    //BFS instead of DFS
    while (stackSize) {
        node = popFirst(&stack, &stackSize);
        // printf("Name: %s\n", node->name);
        if (!contains(visited, nVisited, node)) {
            char* encoded = encodeNode(node, dataFile, &linkLines, &linksCreated);
            push(&visited, &nVisited, node);
            append(&lines, &nLines, encoded);

            for (int i = 0; i < node->m; i++) {
                push(&stack, &stackSize, node->outputs[i]);
            }
        }
    }

    fclose(dataFile);
    *nLinks = linksCreated;
    *nNodeLines = nLines;
    *linkSection = linkLines;
    return lines;
}

int countWhitespace(char* str) {
    int count = 0;
    while (*str) {
        count++;
        str++;
    }
    return count;
}

char** tokenize(char* str) {
    char** tokens = calloc(countWhitespace(str) + 1, sizeof(char*));
    char* tmp = strtok(str, "\n ");
    tokens[0] = calloc(strlen(tmp) + 1, sizeof(char));
    // strcpy(tokens[0], tmp);

    int idx = 0;
    while (tmp) {
        tokens[idx] = calloc(strlen(tmp) + 1, sizeof(char));
        strcpy(tokens[idx], tmp);
        //printf("T: %s\n", tokens[idx]);
        idx++;

        tmp = strtok(NULL, "\n ");
        // printf("tmp: %s\n", tmp);
    }

    return tokens;
}

graph_t* graphFileRead(char* filename) {
    FILE* fp = fopen(filename, "r");

    char buffer[MAX_LINE_LENGTH];
    char graphName[MAX_LINE_LENGTH];
    fgets(graphName, MAX_LINE_LENGTH, fp);
    int nEntryPoints = atoi(fgets(buffer, INT_SIZE, fp));
    int nNodes = atoi(fgets(buffer, INT_SIZE, fp));
    int nLinks = atoi(fgets(buffer, INT_SIZE, fp));

    fgets(buffer, MAX_LINE_LENGTH, fp);
    char* dataFilename = calloc(strlen(buffer) + 1, sizeof(char));
    memcpy(dataFilename, strtok(buffer, "\n"), strlen(buffer));
    data_t** dataArr = readData(dataFilename);

    node_t** nodeArr = calloc(1, sizeof(node_t*));
    int nAdded = 0;

    hashtable_t* nodeNameToId = calloc(1, sizeof(hashtable_t));

    for (int i = 0; i < nNodes; i++) {
        char** tokens = tokenize(fgets(buffer, MAX_LINE_LENGTH, fp));
        node_t* tmp = nodeInit(tokens[1], atoi(tokens[4]), atoi(tokens[5]), atoi(tokens[2]));

        if (tmp->isData) {
            tmp->content.data = dataArr[i];
        } else {
            tmp->content.operation.funcName = decodeOperation(tokens[3]);
        }

        add(nodeNameToId, tokens[1], nAdded);
        push(&nodeArr, &nAdded, tmp);
    }

    for (int i = 0; i < nLinks; i++) {
        char** tokens = tokenize(fgets(buffer, MAX_LINE_LENGTH, fp));
        node_t* input = nodeArr[get(nodeNameToId, tokens[1])];
        node_t* output = nodeArr[get(nodeNameToId, tokens[2])];
        linkNodes(input, output);
    }

    node_t** graphEntryPoints = calloc(nEntryPoints, sizeof(node_t*));
    memcpy(graphEntryPoints, nodeArr, sizeof(node_t*) * nEntryPoints);
    graph_t* returnGraph = malloc(sizeof(graph_t));
    returnGraph->name = graphName;
    returnGraph->n = nEntryPoints;
    returnGraph->entryPoints = graphEntryPoints;

    fclose(fp);
    freeHashtable(nodeNameToId);
    return returnGraph;
}

void graphFileWrite(char* filename, graph_t* graphToWrite) {
    FILE* fp = fopen(filename, "w");

    char line[MAX_LINE_LENGTH];
    sprintf(line, "%s\n%d\n", graphToWrite->name, graphToWrite->n);
    fputs(line, fp);

    int nLines = 0;
    char** linkSection;
    int nLinks = 0;

    char** lines = encodeGraph(graphToWrite, &linkSection, &nLines, &nLinks);

    sprintf(line, "%d\n%d\n", nLines - 1, nLinks);
    fputs(line, fp);

    for (int i = 0; i < nLines; i++, lines++) {
        fputs(*lines, fp);
        fputs("\n", fp);
    }

    for (int i = 0; i < nLinks; i++) {
        fputs(linkSection[i], fp);
    }

    fclose(fp);
}

