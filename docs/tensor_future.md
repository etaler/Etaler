# Design of a full NDArray system

The current Tensor implementation is more like a buffer. Backend can perform specific computation but never the geenral function.

## Design
Each Tensor object hold
1. A shpared_ptr pointing to a TensorImpl object. Which stores the tensor itself. And can be nullptr
2. A shapred_ptr pointing to the parent Tensor object. This is exclusive to the pointer above
3. A Indexer describing how the parent is viewed from the current object.

## OpenCL
JIT the access code

## CPU
Make interpolator