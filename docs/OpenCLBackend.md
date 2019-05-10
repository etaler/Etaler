# About the OpenCL backend

The is a brief document/note about the OpenCL backend.

## OpenCL standard version
Currently the backend only requires and uses the features provided by OpenCL 1.2. But OpenCL 2.0 support is planned.

## Program caching
The OpenCL backend caches programs automatoically. If a program `cast` already exists in the kernel manager. Asking it to compile another version of `cast` does nothing. Set the `force_override` flag to force a compilation and override.

## Using Local Memory
Due to how HTM work - being very memory bangwidth hungry and the input SDR is relativelly small. The OpenCL backend tries to stores data in the GPU's local(on-chip) memory so more bandwidth can be used for fetching synapse.

However this also poses limitations. Since the input Tensor is copied into the local memory. The size of the input Tensor cannot exceed the size of local memory (48KB on NVIDIA cards and 64KB on AMD cards). If larger inputs are encountered, Etaler simply switches to a version not using local memory.
Also some GPUs - all of them are mobile GPUs -  don't have such local memory and is emulated using global memroy. Etaler uses kernels without local memory optimization. 

## OpenCL kenrel distribution
Currently all `.cl` file are stored in the `kernels` folder. But this is not a good idea for software distribution. We'll have to make CMake pack the kernel into .hpp files.

## NVIDIA's OpenCL implementation
NVIDIA's OpenCL implementation can crash without notifing the user. (kerenl can crash without abort, generating error code at the wrong places, etc...). Use POCL's CUDA backend for varification that the kernel is running correctly.

## POCL's CPU backend
POCL 1.3's LLVM-CPU backend seems always crashing.

## program name mangling
Since the OpenCL backend tracks programes using a key. Name mangling (in Etaler's case, appending the hash of the compiler argumnents to the end of the key) is required to support multiple versions of the same program (with different `-D` defines, etc...).

## RPi VC4CL Support
VC4CL is **not suported** for now. The limitation of VC4CL only supporting up to 12 PE per work group is not taken into account in the OpenCL backend. (And VC4 uses global memory to emulate local memory, it is going to be slow),

## Altera AOCL / Xiinx SDAccel support
FPGA based OpenCL although interaseting, are not supported now due to the lack of a API callable compiler.

