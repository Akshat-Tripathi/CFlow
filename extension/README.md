# CFlow - an attempt to recreate TensorFlow

CFlow was developed as an extension to the [C project](https://www.imperial.ac.uk/computing/current-students/computing/computing-first-year/) at the [Department of Computing](https://www.imperial.ac.uk/computing/) at Imperial College London.

## Motivation
We chose this project to gain an understanding of how deep learning models function under the hood and develop our C programming and team working skills.

CFlow attempts to emulate the [Keras Functional API](https://keras.io/guides/functional_api/). What this means is that developers should be able to use CFlow to make neural networks with many input and output heads.

---
## Documentation

#### Graphs
Like Tensorflow, CFlow represents neural networks as computational graphs, with data nodes and operation nodes. Data nodes are used to store both weights and biases and to store input and output data. Operation nodes represent different matrix operations such as: the dot product, convolution, the hadamard product and addition.

Graphs can be created by users, either through programming them, or by writing a .graph file, and reading it with `graphFileRead`. Once a network has finished training, it can be stored as a .graph file while its weights and biases would be stored in an associated .data file. `graphFileWrite` stores the graph node by node using breadth first search, to make it easier for people to read its output.

#### Nodes
As mentioned before, nodes represent either data (in the form of matrices) or matrix operations. You may notice that nodes have the matrices: `poolingMatrixGrad` and `optimiserMatrix`, these are used during backpropagation. PoolingMatrixGrad is used to store the indices used during max pooling. OptimiserMatrix is used by optimisers to store the gradient accumulations during training.

Nodes also store pointers to their input and output nodes, which allow graphs to be traversed bidirectionally.

#### Matrices and Operations

