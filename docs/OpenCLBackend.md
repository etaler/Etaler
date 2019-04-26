# About the OpenCL backend

The is a brief document/note about the OpenCL backend.

## Program caching
The OpenCL backend caches programs automatoically. If a program `cast` already exists in the kernel manager. Asking it to compile another version of `cast` will do nothing. Use the `force_override` flag to force a override.

## Local Memory usage
Due to how HTM work - being very memory bangwidth hungry and the input SDR is relativelly small. The OpenCL backend tries to stores data in the GPU's local memory so more bandwidth can be used for fetching synapse.

But this also posted some limitation. Since input Tensor is stored into the local memory. The size of the input Tensor cannot exceed the size of local memory (48KB on NVIDIA cards and 64KB on AMD cards). This is limitation will be removed in future versions. But not using local memory will come with a huge performance panality.

## OpenCL kenrel distribution
Currently all `.cl` file are stored in the `kerenls` folder. But this is not a good idea for softward distribution. We'll have to make CMake pack the kernel into .hpp files.

## NVIDIA's OpenCL implementation
NVIDIA's OpenCL implementation can crash without notifing the user. (kerenl can crash without abort, generating error code, etc...). Use POCL's CUDA backend for varification that the kernel is running correctly.

## POCL's CPU backend
POCL 1.3's LLVM-CPU backend seems always crashing.