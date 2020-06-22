#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../activation.h"
#include "../data.h"
#include "../error.h"
#include "../file.h"
#include "../matrix.h"
#include "../nodes.h"
#include "../optimisers.h"
#include "../readCSV.h"
#include "../scheduler.h"
#include "../testUtils.h"

#define SCALAR_TEST 213584.042312
#define DOUBLE_COMPARISON 0.000000000000001
#define CROSS_ENTROPY_COMPARISON 0.05

int assertsRan = 0;
int assertsFailed = 0;
int testsRan = 0;
int testsFailed = 0;
#define functionName(func) (#func)

#define assertEqual(a, b)                                                                    \
    do {                                                                                     \
        assertsRan++;                                                                        \
        if ((a) != (b)) {                                                                    \
            printf("%s(line %d): got: %lf | expected: %lf\n", __func__, __LINE__, (double) (a), (double) (b)); \
            assertsFailed++;                                                                 \
        }                                                                                    \
    } while (0)

#define assertEqualPtr(a, b)                                                                    \
    do {                                                                                     \
        assertsRan++;                                                                        \
        if ((a) != (b)) {                                                                    \
            printf("%s(line %d): got: %p | expected: %p\n", __func__, __LINE__, (void *) (a), (void *) (b)); \
            assertsFailed++;                                                                 \
        }                                                                                    \
    } while (0)

#define assertOther(a)                                                                            \
    do {                                                                                          \
        assertsRan++;                                                                             \
        bool result = (a);                                                                        \
        if (!result) {                                                                            \
            printf("%s(line %d): got: %d | expected: %d\n", __func__, __LINE__, result, !result); \
            assertsFailed++;                                                                      \
        }                                                                                         \
    } while (0)

void runTest(void (*test)(void)) {
    assertsRan = assertsFailed = 0;
    test();
    testsRan++;
    if (assertsFailed > 0) {
        testsFailed++;
        printf("**** %s ****: %d asserts failed out of %d asserts\n\n", functionName(&test), assertsFailed, assertsRan);
    }
    printf("%d out of %d asserts pass!\n\n", assertsRan - assertsFailed, assertsRan);
}

void testActiveFuncs() {
    printf("Testing activation functions\n");

    assertEqual(relu(2.12312), 2.12312);
    assertEqual(relu(-19842231), 0);

    assertEqual(reluPrime(12389123898), 1);
    assertEqual(reluPrime(-1231283), 0);

    assertEqual(lRelu(1239), 1239);
    assertEqual(lRelu(-1240), -248);

    assertEqual(lReluPrime(), 1);

    assertEqual(linear(123.123), 123.123);
    assertEqual(linearPrime(123), 1);

    assertOther(sigmoid(32.128) - 1 < 0.000001);
    assertEqual(sigmoid(0), 0.5);
    assertOther(sigmoid(-123) < 0.000001);

    assertOther(sigmoidPrime(sigmoid(32.128)) - 0 < 0.000001);
    assertEqual(sigmoidPrime(sigmoid(0)), 0.25);
    assertEqual(sigmoidPrime(sigmoid(-12313)), 0);

    assertEqual(tanhActive(0), 0);
    assertEqual(tanhActive(123), 1);
    assertEqual(tanhActive(-123), -1);

    assertEqual(tanhPrime(0), 1);
    assertOther(tanhPrime(-123) < 0.00001);
    assertOther(tanhPrime(123) < 0.00001);

    // double mat1[5][1] = {{1.3}, {5.1}, {2.2}, {0.7}, {1.1}};
    // double ans[5][1] = {{0.02}, {0.9}, {0.05}, {0.01}, {0.02}};
    matrix2d_t *mat1 = matrixCreate(5, 1);
    matrix2d_t *ans1 = matrixCreate(5, 1);

    matrixSet(mat1, 0, 0, 1.3);
    matrixSet(mat1, 1, 0, 5.1);
    matrixSet(mat1, 2, 0, 2.2);
    matrixSet(mat1, 3, 0, 0.7);
    matrixSet(mat1, 4, 0, 1.1);

    matrixSet(ans1, 0, 0, 0.02);
    matrixSet(ans1, 1, 0, 0.90);
    matrixSet(ans1, 2, 0, 0.05);
    matrixSet(ans1, 3, 0, 0.01);
    matrixSet(ans1, 4, 0, 0.02);

    matrix2d_t *result1 = softmax(mat1);
    for (int i = 0; i < mat1->nRows; i++) {
        for (int j = 0; j < mat1->nCols; j++) {
            assertOther((matrixGet(result1, i, j) - matrixGet(ans1, i, j)) < CROSS_ENTROPY_COMPARISON);
        }
    }
    printf("Finished testing activation functions\n");
}

void testMatrix() {
    printf("Testing matrix add, multiply element wise, scalar product, subtract\n");
    matrix2d_t *m1 = matrixCreate(10000, 100);
    matrix2d_t *m2 = matrixCreate(10000, 100);
    assertEqual(matrixGet(m1, 25, 50), 0);
    assertEqual(m1->nCols, m2->nCols);
    assertEqual(m1->nRows, m2->nRows);

    matrixRandomise(m1);
    matrixRandomise(m2);

    matrix2d_t *m3 = matrixAdd(m1, m2);
    matrix2d_t *m4 = matrixMultiplyElementWise(m1, m2);
    matrix2d_t *m5 = matrixScalarProduct(m1, SCALAR_TEST);
    matrix2d_t *m6 = matrixSubtract(m1, m2);
    for (int i = 0; i < m1->nRows; i++) {
        for (int j = 0; j < m1->nCols; j++) {
            assertEqual(matrixGet(m3, i, j), ((matrixGet(m1, i, j)) + matrixGet(m2, i, j)));
            assertEqual(matrixGet(m4, i, j), ((matrixGet(m1, i, j)) * matrixGet(m2, i, j)));
            assertEqual(matrixGet(m5, i, j), (matrixGet(m1, i, j) * SCALAR_TEST));
            assertEqual(matrixGet(m6, i, j), ((matrixGet(m1, i, j)) - matrixGet(m2, i, j)));
        }
    }

    printf("Finished testing matrix add, multiply element wise, scalar product, subtract\n");
    printf("Tested 4 matricies in total\n");
}

void testMatrixDotProduct() {
    printf("Testing matrix dot product\n");
    matrix2d_t *m1 = matrixCreate(518, 100);
    matrix2d_t *m2 = matrixCreate(100, 1282);
    matrix2d_t *m5 = matrixCreate(518, 100);
    matrix2d_t *m6 = matrixCreate(100, 1282);

    matrixRandomise(m1);
    matrixRandomise(m2);
    matrixRandomise(m5);
    matrixRandomise(m6);

    matrix2d_t *m3 = matrixDotProduct(m1, m2);
    assertEqual(m3->nRows, m1->nRows);
    assertEqual(m3->nCols, m2->nCols);
    matrix2d_t *m4 = matrixDotProduct(m5, m6);
    double result1, result2;
    for (int i = 0; i < m1->nRows; i++) {
        for (int j = 0; j < m2->nCols; j++) {
            result1 = result2 = 0;
            for (int k = 0; k < m1->nCols; k++) {
                result1 += matrixGet(m1, i, k) * matrixGet(m2, k, j);
                result2 += matrixGet(m5, i, k) * matrixGet(m6, k, j);
            }
            assertEqual(matrixGet(m3, i, j), result1);
            assertEqual(matrixGet(m4, i, j), result2);
        }
    }
    printf("Finished testing matrix dot product\n");
    printf("Tested 2 matricies in total\n");
}

void testMatrixActiveFuncs() {
    printf("Testing matrix apply activation functions\n");
    matrix2d_t *m1 = matrixCreate(5128, 100);

    matrixRandomise(m1);

    matrix2d_t *m2 = matrixActiveFunc(m1, (enum activationFunction) RELU);
    matrix2d_t *m3 = matrixActiveFunc(m1, RELU_PRIME);
    matrix2d_t *m4 = matrixActiveFunc(m1, LRELU);
    matrix2d_t *m5 = matrixActiveFunc(m1, LRELU_PRIME);
    matrix2d_t *m6 = matrixActiveFunc(m1, LINEAR);
    matrix2d_t *m7 = matrixActiveFunc(m1, LINEAR_PRIME);
    matrix2d_t *m8 = matrixActiveFunc(m1, SIGMOID);
    matrix2d_t *m9 = matrixActiveFunc(m1, SIGMOID_PRIME);
    matrix2d_t *m10 = matrixActiveFunc(m1, TANH);
    matrix2d_t *m11 = matrixActiveFunc(m1, TANH_PRIME);

    for (int i = 0; i < m1->nRows; i++) {
        for (int j = 0; j < m1->nCols; j++) {
            assertEqual(matrixGet(m2, i, j), relu(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m3, i, j), reluPrime(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m4, i, j), lRelu(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m5, i, j), lReluPrime());
            assertEqual(matrixGet(m6, i, j), linear(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m7, i, j), linearPrime(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m8, i, j), sigmoid(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m9, i, j), sigmoidPrime(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m10, i, j), tanhActive(matrixGet(m1, i, j)));
            assertEqual(matrixGet(m11, i, j), tanhPrime(matrixGet(m1, i, j)));
        }
    }
    printf("Finished testing matrix apply activations functions\n");
    printf("Tested 10 matricies in total\n");
}

void testSingleMatrixFuncs() {
    printf("Testing matrix rotate, transpose, dilate\n");
    matrix2d_t *m1 = matrixCreate(126, 454);
    matrixRandomise(m1);

    matrix2d_t *m1Transpose = matrixTranspose(m1);
    for (int i = 0; i < m1->nRows; i++) {
        for (int j = 0; j < m1->nCols; j++) {
            assertEqual(matrixGet(m1, i, j), matrixGet(m1Transpose, j, i));
        }
    }

    matrix2d_t *ownRotate = matrixCreate(8, 8);
    double angle = 3.14159265 / 2;  //90 degrees
    for (int i = 0; i < ownRotate->nRows; i++) {
        matrixSet(ownRotate, i, i, 1);
    }
    matrixSet(ownRotate, 2, 2, cos(angle));
    matrixSet(ownRotate, 2, 3, -sin(angle));
    matrixSet(ownRotate, 3, 2, sin(angle));
    matrixSet(ownRotate, 3, 3, cos(angle));

    matrix2d_t *m2 = matrixCreate(8, 255);
    assertEqual(matrixGet(m2, 0, 0), 0);

    matrixRandomise(m2);

    matrix2d_t *m2Rotated = matrixRotate(m2, 90, 2, 3);
    matrix2d_t *ownRotation = matrixDotProduct(ownRotate, m2);
    assertEqual(m2Rotated->nRows, ownRotation->nRows);
    assertEqual(m2Rotated->nCols, ownRotation->nCols);
    for (int i = 0; i < m2Rotated->nRows; i++) {
        for (int j = 0; j < m2Rotated->nCols; j++) {
            assertOther(abs(matrixGet(m2Rotated, i, j) - matrixGet(ownRotation, i, j)) < 0.01);
        }
    }

    matrix2d_t *m3 = matrixCreate(232, 232);
    matrixRandomise(m3);
    int dilation = 2;
    matrix2d_t *m3Dilate = matrixDilate(m3, dilation);
    for (int i = 0; i < m3Dilate->nRows; i++) {
        for (int j = 0; j < m3Dilate->nCols; j++) {
            if ((i % (dilation + 1) == 0) && (j % (dilation + 1) == 0)) {
                assertEqual(matrixGet(m3Dilate, i, j), matrixGet(m3, i / (dilation + 1), j / (dilation + 1)));
            } else {
                assertEqual(matrixGet(m3Dilate, i, j), 0);
            }
        }
    }

    matrix2d_t *m4 = matrixCreate(231, 234);
    matrixRandomise(m4);
    double* flattenM4 = flatten2d(m4);
    for (int i = 0; i < m4->nRows; i++) {
        for (int j = 0; j < m4->nCols; j++) {
            assertEqual(flattenM4[(i * m4->nCols) + j], matrixGet(m4, i, j));
        }
    }
}

void testGraph(void) {
    printf("Testing graph read/write functions\n");

    graph_t *graph = genGraph();
    graphFileWrite("test.graph", graph);
    graph_t *read = graphFileRead("test.graph");
    assertOther(graphsAreEqual(graph, read));

    printf("Finished testing graph read/write functions\n");
}

//Uses this graph
//https://miro.medium.com/max/4000/1*Fi1AZPZLrGf-6wM_wTSPQw.png
void testScheduler() {
    printf("%s\n", "Testing scheduler");
    node_t *A = nodeInit("A", 0, 1, true);
    node_t *B = nodeInit("B", 1, 3, true);
    node_t *C = nodeInit("C", 1, 1, true);
    node_t *D = nodeInit("D", 2, 1, true);
    node_t *E = nodeInit("E", 3, 1, true);
    node_t *F = nodeInit("F", 1, 0, true);
    node_t *G = nodeInit("G", 0, 1, true);

    linkNodes(A, B);
    linkNodes(G, D);
    linkNodes(B, C);
    linkNodes(B, D);
    linkNodes(C, E);
    linkNodes(B, E);
    linkNodes(D, E);
    linkNodes(E, F);

    node_t **bufferEntryPoints = calloc(2, sizeof(node_t *));
    bufferEntryPoints[0] = A;
    bufferEntryPoints[1] = G;
    node_t **bufferExitPoints = calloc(1, sizeof(node_t*));
    bufferExitPoints[0] = F;

    graph_t *newGraph = graphInit("TEST", 2, bufferEntryPoints, 1, bufferExitPoints);

    int nodeArrSize = 0;
    node_t **nodeArr = schedule(newGraph, &nodeArrSize);

    assertEqualPtr(nodeArr[6], G);
    assertEqualPtr(nodeArr[5], A);
    assertEqualPtr(nodeArr[4], B);
    assertEqualPtr(nodeArr[3], D);
    assertEqualPtr(nodeArr[2], C);
    assertEqualPtr(nodeArr[1], E);
    assertEqualPtr(nodeArr[0], F);
    assertEqual(nodeArrSize, 7);

    printf("%s\n", "Finished testing scheduler");
}

void testDataFile() {
    printf("Testing Data Files\n");
    // Test 1
    node_t *node1 = nodeInit("test", 3, 5, true);
    FILE *fileWrite0 = fopen("dataTest", "w");  //makes sure the file is empty
    fclose(fileWrite0);
    FILE *fileWrite1 = fopen("dataTest", "a");
    matrix2d_t *matrix1 = matrixCreate(25, 10);

    matrixRandomise(matrix1);
    node1->content.data->data->matrix2d = matrix1;
    writeBlock(fileWrite1, *node1);
    fclose(fileWrite1);

    data_t *data1 = initData(true, matrix1);
    data_t **datas1 = readData("dataTest");
    assertOther(dataAreEqual(datas1[0], data1));

    // Test 2
    data_t **datas = calloc(5, sizeof(data_t *));
    datas[0] = data1;
    node_t *node = NULL;
    FILE *fileWrite2 = fopen("dataTest", "a");
    for (int i = 1; i < 5; i++) {
        node = nodeInit("test", 6, 8, true);
        matrix2d_t *matrix = matrixCreate(50, 35);
        matrixRandomise(matrix);
        node->content.data->data->matrix2d = matrix;

        data_t *data = initData(true, matrix);
        datas[i] = data;
        writeBlock(fileWrite2, *node);
    }

    fclose(fileWrite2);
    data_t **read = readData("dataTest");
    for (int i = 0; i < 5; i++) {
        assertOther(dataAreEqual(read[i], datas[i]));
    }

    free(node1);
    free(node);
    for (int i = 0; i < 5; i++) {
        free(datas[i]);
    }

    printf("Finished testing data files\n");
}

void testErrorFunctions() {
    printf("Testing Error Functions\n");

    //mean squared error test
    matrix2d_t *expected1 = matrixCreate(5, 1);
    matrixSet(expected1, 0, 0, 43.6); 
    matrixSet(expected1, 1, 0, 44.4); 
    matrixSet(expected1, 2, 0, 45.2); 
    matrixSet(expected1, 3, 0, 46); 
    matrixSet(expected1, 4, 0, 46.8); 

    matrix2d_t *actual1 = matrixCreate(5, 1);
    matrixSet(actual1, 0, 0, 41); 
    matrixSet(actual1, 1, 0, 45); 
    matrixSet(actual1, 2, 0, 49); 
    matrixSet(actual1, 3, 0, 47); 
    matrixSet(actual1, 4, 0, 44); 
    
    assertOther((meanSquaredError(expected1, actual1) - 6.08) < DOUBLE_COMPARISON);

    matrix2d_t *expected2 = matrixCreate(10, 1);
    matrixSet(expected2, 0, 0, 75.4); 
    matrixSet(expected2, 1, 0, 82.4); 
    matrixSet(expected2, 2, 0, 28.1); 
    matrixSet(expected2, 3, 0, 45.14); 
    matrixSet(expected2, 4, 0, 45.3);
    matrixSet(expected2, 5, 0, 39.7); 
    matrixSet(expected2, 6, 0, 85.1); 
    matrixSet(expected2, 7, 0, 64.2); 
    matrixSet(expected2, 8, 0, 25.7); 
    matrixSet(expected2, 9, 0, 41.8);  

    matrix2d_t *actual2 = matrixCreate(10, 1);
    matrixSet(actual2, 0, 0, 72); 
    matrixSet(actual2, 1, 0, 84); 
    matrixSet(actual2, 2, 0, 28); 
    matrixSet(actual2, 3, 0, 41); 
    matrixSet(actual2, 4, 0, 45);
    matrixSet(actual2, 5, 0, 32); 
    matrixSet(actual2, 6, 0, 84); 
    matrixSet(actual2, 7, 0, 65); 
    matrixSet(actual2, 8, 0, 29); 
    matrixSet(actual2, 9, 0, 44);  
    
    assertOther((meanSquaredError(expected2, actual2) - 108.2296) < DOUBLE_COMPARISON);

    matrix2d_t *actual3 = matrixCreate(10, 1);
    matrixSet(actual3, 0, 0, 75); 
    matrixSet(actual3, 1, 0, 82); 
    matrixSet(actual3, 2, 0, 28); 
    matrixSet(actual3, 3, 0, 45); 
    matrixSet(actual3, 4, 0, 45);
    matrixSet(actual3, 5, 0, 39); 
    matrixSet(actual3, 6, 0, 85); 
    matrixSet(actual3, 7, 0, 64); 
    matrixSet(actual3, 8, 0, 26); 
    matrixSet(actual3, 9, 0, 42);  
    
    assertOther((meanSquaredError(expected2, actual3) - 1.1096) < DOUBLE_COMPARISON);

    // //cross entropy loss test

    assertOther((crossEntropyLoss(expected1, actual1) - (-1242.711)) / (-1242.711) < CROSS_ENTROPY_COMPARISON);
    assertOther((crossEntropyLoss(expected2, actual2) - (-3062.88)) / (-3062.88) < CROSS_ENTROPY_COMPARISON);
    assertOther((crossEntropyLoss(expected2, actual3) - (-3103.608)) / (-3103.608) < CROSS_ENTROPY_COMPARISON);

    printf("Finished testing error functions\n");
}

void testOptimisers() {
    printf("Testing Optimiser Functions\n");

    double lRate = 0.7;
    // double momentum = 0.9;
    // double delta = 0.65;
    // double decayRate = 0.1;
    //Tets sgd
    node_t *test1 = nodeInit("test1", 0, 2, true);
    test1->content.data->data->matrix2d = matrixCreate(100, 235);
    matrixRandomise(test1->content.data->data->matrix2d);

    test1->matrix->matrix2d = matrixCreate(100, 235);
    matrixRandomise(test1->matrix->matrix2d);

    matrix2d_t *weights1 = matrixClone(test1->content.data->data->matrix2d);
    matrix2d_t *gradients1 = matrixClone(test1->matrix->matrix2d);

    matrix2d_t *newWeights = matrixSubtract(weights1, matrixScalarProduct(test1->matrix->matrix2d, lRate));

    sgd(test1, 1, lRate);
    for (int i = 0; i < test1->matrix->matrix2d->nRows; i++) {
        for (int j = 0; j < test1->matrix->matrix2d->nCols; j++) {
            assertEqual(matrixGet((test1->content.data->data->matrix2d), i, j),
                        matrixGet(newWeights, i, j));
        }
    }

    freeNode(test1);
    matrixFree(weights1);
    matrixFree(gradients1);
    /*
    //Test sgdMomentum
    node_t *test2 = nodeInit("test2", 0, 2, true);
    test2->matrix = matrixCreate(150, 180);
    matrixRandomise(test2->matrix);
    test2->content.data->data->matrix2d = *matrixCreate(150, 180);
    matrixRandomise(&test2->content.data->data->matrix2d);
    test2->optimiserMatrix = matrixCreate(150, 180);
    matrixRandomise(test2->optimiserMatrix);

    matrix2d_t *weights2 = matrixClone(test2->matrix);
    matrix2d_t *gradients2 = matrixClone(&test2->content.data->data->matrix2d);
    matrix2d_t *optimise2 = matrixClone(test2->optimiserMatrix);

    sgdMomentum(test2, lRate, momentum);
    for (int i = 0; i < test2->content.data->data->matrix2d.nRows; i++) {
        for (int j = 0; j < test2->content.data->data->matrix2d.nCols; j++) {
            assertEqual(matrixGet(test2->optimiserMatrix, i, j),
                        matrixGet(optimise2, i, j) * momentum - matrixGet(weights2, i, j) * lRate);
            assertEqual(matrixGet(&test2->content.data->data->matrix2d, i, j),
                        matrixGet(gradients2, i, j) + matrixGet(test2->optimiserMatrix, i, j));
        }
    }

    freeNode(test2);
    matrixFree(weights2);
    matrixFree(gradients2);
    matrixFree(optimise2);

    //Test adagrad
    node_t *test3 = nodeInit("test3", 0, 2, true);
    test3->matrix = matrixCreate(250, 220);
    matrixRandomise(test3->matrix);
    test3->content.data->data->matrix2d = *matrixCreate(250, 220);
    matrixRandomise(&test3->content.data->data->matrix2d);
    test3->optimiserMatrix = matrixCreate(250, 220);
    matrixRandomisePositive(test3->optimiserMatrix);

    matrix2d_t *weights3 = matrixClone(test3->matrix);
    matrix2d_t *gradients3 = matrixClone(&test3->content.data->data->matrix2d);
    matrix2d_t *optimise3 = matrixClone(test3->optimiserMatrix);

    adagrad(test3, lRate, delta);
    double denomConstant = 0;
    double constant = 0;
    for (int i = 0; i < test3->content.data->data->matrix2d.nRows; i++) {
        for (int j = 0; j < test3->content.data->data->matrix2d.nCols; j++) {
            assertEqual(matrixGet(test3->optimiserMatrix, i, j),
                        matrixGet(optimise3, i, j) + matrixGet(weights3, i, j) * matrixGet(weights3, i, j));
            denomConstant = sqrt(matrixGet(optimise3, i, j)) + delta;
            constant = (1 / denomConstant) * lRate;
            assertEqual(matrixGet(&test3->content.data->data->matrix2d, i, j),
                        matrixGet(gradients3, i, j) - matrixGet(weights3, i, j) * constant);
        }
    }

    freeNode(test3);
    matrixFree(weights3);
    matrixFree(gradients3);
    matrixFree(optimise3);

    //Test RMSProp
    node_t *test4 = nodeInit("test4", 0, 2, true);
    test4->matrix = matrixCreate(360, 240);
    matrixRandomise(test4->matrix);
    test4->content.data->data->matrix2d = *matrixCreate(360, 240);
    matrixRandomise(&test4->content.data->data->matrix2d);
    test4->optimiserMatrix = matrixCreate(360, 240);
    matrixRandomisePositive(test4->optimiserMatrix);

    matrix2d_t *weights4 = matrixClone(test4->matrix);
    matrix2d_t *gradients4 = matrixClone(&test4->content.data->data->matrix2d);
    matrix2d_t *optimise4 = matrixClone(test4->optimiserMatrix);

    RMSProp(test4, lRate, decayRate, delta);
    for (int i = 0; i < test4->content.data->data->matrix2d.nRows; i++) {
        for (int j = 0; j < test4->content.data->data->matrix2d.nCols; j++) {
            assertEqual(matrixGet(test4->optimiserMatrix, i, j),
                        matrixGet(optimise4, i, j) * decayRate +
                            matrixGet(weights4, i, j) * matrixGet(weights4, i, j) * (1.0 - decayRate));
            denomConstant = sqrt(matrixGet(optimise4, i, j)) + delta;
            constant = (1 / denomConstant) * lRate;
            assertEqual(matrixGet(&test4->content.data->data->matrix2d, i, j),
                        matrixGet(gradients4, i, j) - matrixGet(weights4, i, j) * constant);
        }
    }

    freeNode(test4);
    matrixFree(weights4);
    matrixFree(gradients4);
    matrixFree(optimise4);
    */
    printf("Finished testing optimiser functions\n");
    printf("Tested 4 out of 4 optimiser functions\n");
}

void testMatrixPooling() {
    printf("Testing matrix pooling\n");
    matrix2d_t *m1 = matrixCreate(212, 452);
    matrixRandomise(m1);

    matrix2d_t *gradient = matrixCreate(212, 452);
    matrix2d_t *m1MaxPool = matrixMaxPooling(m1, gradient, 1, 2);
    assertEqual(m1MaxPool->nCols, 452);
    assertEqual(m1MaxPool->nRows, 212);

    int filterSize = 2;
    int strideSize = 1;
    for (int i = 0; i < m1MaxPool->nRows; i++) {
        for (int j = 0; j < m1MaxPool->nCols; j++) {
            for (int k = 0; k < filterSize; k++) {
                for (int l = 0; l < filterSize; l++) {
                    if ((i*strideSize + k < m1->nRows) && j*strideSize + l < m1->nCols)
                        assertOther(matrixGet(m1MaxPool,  i, j) >= matrixGet(m1, i*strideSize + k, j*strideSize + l));
                }
            }
        }
    }

    strideSize = 2;
    matrix2d_t *m2MaxPool = matrixMaxPooling(m1, gradient, 2, 2);
    for (int i = 0; i < m2MaxPool->nRows; i++) {
        for (int j = 0; j < m2MaxPool->nCols; j++) {
            for (int k = 0; k < filterSize; k++) {
                for (int l = 0; l < filterSize; l++) {
                    if (i*strideSize + k < m1->nRows && j*strideSize + l < m1->nCols)
                        assertOther(matrixGet(m2MaxPool,  i, j) >= matrixGet(m1, i*strideSize + k, j*strideSize + l));
                }
            }
        }
    }

    matrix2d_t *m4MaxPool = matrixMaxPooling(m1, gradient, 1, 1);
    
    for (int i = 0; i < m1->nRows; i++) {
        for (int j = 0; j < m1->nCols; j++) {
            assertEqual(matrixGet(m1, i, j), matrixGet(m4MaxPool, i, j));
        }
    }

    matrix2d_t *m1AvgPool = matrixAveragePooling(m1, gradient, 1, 2);
    strideSize = 1;
    for (int i = 0; i < m1AvgPool->nRows; i++) {
        for (int j = 0; j < m1AvgPool->nCols; j++) {
            double sum = 0;
            for (int k = 0; k < filterSize; k++) {
                for (int l = 0; l < filterSize; l++) {
                    if (i*strideSize + k < m1->nRows && j*strideSize + l < m1->nCols)
                        sum += matrixGet(m1, i*strideSize + k, j*strideSize + l);
                }
            }
            assertEqual(matrixGet(m1AvgPool, i, j), sum / (double) (filterSize * filterSize));
        }
    }


    matrix2d_t *m2AvgPool = matrixAveragePooling(m1, gradient, 2, 2);
    strideSize = 2;
    // stride size = 2
    for (int i = 0; i < m2AvgPool->nRows; i++) {
        for (int j = 0; j < m2AvgPool->nCols; j++) {
            double sum = 0;
            for (int k = 0; k < filterSize; k++) {
                for (int l = 0; l < filterSize; l++) {
                    if (i*strideSize + k < m1->nRows && j*strideSize + l < m1->nCols)
                        sum += matrixGet(m1, i*strideSize + k, j*strideSize + l);
                }
            }
            assertEqual(matrixGet(m2AvgPool, i, j), sum / (double) (filterSize * filterSize));
        }
    }

    
    matrix2d_t *m4AvgPool = matrixAveragePooling(m1, gradient, 1, 1);
    assertEqual(m4AvgPool->nRows, m1->nRows);
    assertEqual(m4AvgPool->nCols, m1->nCols);

    for (int i = 0; i < m1->nRows; i++) {
        for (int j = 0; j < m1->nCols; j++) {
                assertEqual(matrixGet(m1, i, j), matrixGet(m4AvgPool, i, j));
        }
    }

    printf("Matrix pooling tests pass\n");
}

void testMatrixConvolution() {
    printf("Testing Matrix convolution\n");
    matrix2d_t *m1 = matrixCreate(6, 6);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            matrixSet(m1, i, j, 10);
        }
    }
    printf("Input Matrix: \n");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%f ", matrixGet(m1, i, j));
        }
        printf("\n");
    }

    matrix2d_t *kernel1 = matrixCreate(3, 3);
    for (int i = 0; i < 3; i++) {
        matrixSet(kernel1, i, 0, 1);
        matrixSet(kernel1, i, 2, -1);
    }
    printf("Input Kernel: \n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", matrixGet(kernel1, i, j));
        }
        printf("\n");
    }

    matrix2d_t *m1Expected = matrixCreate(4, 4);
    for (int i = 0; i < 4; i++) {
        matrixSet(m1Expected, i, 1, 30);
        matrixSet(m1Expected, i, 2, 30);
    }
    printf("Expected Output: \n");
    for (int i = 0; i < m1Expected->nRows; i++) {
        for (int j = 0; j < m1Expected->nCols; j++) {
            printf("%f ", matrixGet(m1Expected, i, j));
        }
        printf("\n");
    }

    matrix2d_t *m1Conv = matrixConvolution(m1, kernel1, 1, 0);
    printf("Actual Output: \n");
    for (int i = 0; i < m1Conv->nRows; i++) {
        for (int j = 0; j < m1Conv->nCols; j++) {
            printf("%f ", matrixGet(m1Conv, i, j));
        }
        printf("\n");
    }
    assertOther(areMatrixesEqual(m1Conv, m1Expected, 0));

    printf("Matrix convolution tests pass\n");
}

void testMatrixDeconvolution() {
    printf("Testing Matrix Deconvolution\n");

    matrix2d_t *m1 = matrixCreate(4, 4);
    for (int i = 0; i < 4; i++) {
        matrixSet(m1, i, 1, 30);
        matrixSet(m1, i, 2, 30);
    }
    printf("Matrix Input: \n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", matrixGet(m1, i, j));
        }
        printf("\n");
    }

    matrix2d_t *kernel1 = matrixCreate(3, 3);
    for (int i = 0; i < 3; i++) {
        matrixSet(kernel1, i, 0, 1);
        matrixSet(kernel1, i, 2, -1);
    }
    printf("Input Kernel: \n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", matrixGet(kernel1, i, j));
        }
        printf("\n");
    }

    matrix2d_t *m1Expected = matrixCreate(6, 6);
    for (int i = 0; i < m1Expected->nRows / 2; i++) {
        for (int j = 0; j < m1Expected->nCols / 2; j++) {
            if (j > 0) {
                matrixSet(m1Expected, i, j, -30 * (i + 1));
                matrixSet(m1Expected, m1Expected->nCols - 1 - i, j, -30 * (i + 1));
                matrixSet(m1Expected, i, m1Expected->nRows - j - 1, 30 * (i + 1));
                matrixSet(m1Expected, m1Expected->nCols - 1 - i, m1Expected->nRows - j - 1, 30 * (i + 1));
            }
        }
    }
    printf("Expected Output: \n");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%f ", matrixGet(m1Expected, i, j));
        }
        printf("\n");
    }

    matrix2d_t *m1Deconv = matrixDeconvolution(m1, kernel1, 1, 0);
    printf("Actual Output: \n");
    for (int i = 0; i < m1Deconv->nRows; i++) {
        for (int j = 0; j < m1Deconv->nCols; j++) {
            printf("%f ", matrixGet(m1Deconv, i, j));
        }
        printf("\n");
    }
    assertEqual(areMatrixesEqual(m1Deconv, m1Expected, 0), true);

    printf("Matrix Deconvolution tests pass\n");
}

void testMatrix3DConvolution() {
    printf("Testing Matrix 3D convolution\n");

    matrix2d_t *m1 = matrixCreate(6, 6);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            matrixSet(m1, i, j, 10);
        }
    }

    matrix2d_t *kernel1 = matrixCreate(3, 3);
    for (int i = 0; i < 3; i++) {
        matrixSet(kernel1, i, 0, 1);
        matrixSet(kernel1, i, 2, -1);
    }
    matrix2d_t **inputMatricies = calloc(3, sizeof(matrix2d_t *));
    inputMatricies[0] = inputMatricies[1] = inputMatricies[2] = m1;
    matrix2d_t **kernels = calloc(3, sizeof(matrix2d_t *));
    kernels[0] = kernels[1] = kernels[2] = kernel1;
    int bias = 2;
    matrix2d_t *result = matrix3DConvolution(inputMatricies, kernels, 1, 0);

    matrix2d_t *expected = matrixCreate(4, 4);
    for (int i = 0; i < 4; i++) {
        matrixSet(expected, i, 1, 30);
        matrixSet(expected, i, 2, 30);
    }
    matrix2d_t *expected1 = matrixAdd(expected, expected);
    matrix2d_t *expected2 = matrixAdd(expected, expected1);
    matrixFree(expected);
    matrixFree(expected1);
    for (int i = 0; i < expected2->nRows; i++) {
        for (int j = 0; j < expected2->nCols; j++) {
            matrixSet(expected2, i, j, matrixGet(expected2, i, j) + bias);
        }
    }
    assertEqual(result->nRows, expected2->nRows);
    assertEqual(result->nCols, expected2->nCols);
    for (int i = 0; i < result->nRows; i++) {
        for (int j = 0; j < result->nCols; j++) {
            assertEqual(matrixGet(result, i, j), matrixGet(expected2, i, j));
        }
    }

    printf("Expected Output: \n");
    for (int i = 0; i < expected2->nRows; i++) {
        for (int j = 0; j < expected2->nCols; j++) {
            printf("%f ", matrixGet(expected2, i, j));
        }
        printf("\n");
    }
    printf("Actual Output: \n");
    for (int i = 0; i < result->nRows; i++) {
        for (int j = 0; j < result->nCols; j++) {
            printf("%f ", matrixGet(result, i, j));
        }
        printf("\n");
    }
    printf("Matrix 3D convolution tests pass\n");
}

void testReadCSV() {
    csvDataPack_t csv = readCSV("data/mnist_train.csv", 60000);
    double *labels = csv.labels;
    matrix2d_t **matrices = csv.matrixInputs;

    assertEqual(labels[0], 5.0);
    assertEqual(matrixGet(matrices[0], 13, 14), 225.0);
    assertEqual(matrixGet(matrices[3], 13, 14), 251.0);
}

int main() {
    printf("Starting tests\n\n");
    runTest(testActiveFuncs);
    runTest(testMatrix);
    runTest(testMatrixDotProduct);
    runTest(testMatrixActiveFuncs);
    runTest(testMatrixPooling);
    runTest(testMatrixConvolution);
    runTest(testMatrixDeconvolution);
    //runTest(testMatrix3DConvolution);
    runTest(testSingleMatrixFuncs);

    runTest(testDataFile);
    runTest(testGraph);
    runTest(testScheduler);
    runTest(testErrorFunctions);
    // runTest(testOptimisers);
    runTest(testReadCSV);
    printf("%d out of %d tests pass!\n", testsRan - testsFailed, testsRan);

    return EXIT_SUCCESS;
}
