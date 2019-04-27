
# Etaler

---

Etaler is a library for machine intelligence based on [HTM theory](https://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf). Providing two main features.

* HTM algorithms with modern API design
* A minimal cross-platform (CPU, GPU, etc..) Tensor implementation

You can now explore HTM with modern, easy to use API and enjoy the performance boost by the GPU.

## More about Etaler

### A GPU ready HTM library
Unlike most previous HTM implementations, Etaler is designed from the ground up to work with GPUs and allows almost seamless data transfer between CPU and GPUs.

Etaler provides HTM algorithms and a minimal Tensor implementation that operates on both CPU and GPU. You can select what works best for you and switch between them if you wish so with ease.

### Total front-end /back-end separation
Etaler is written in a way that the front-end (Tensor operation calls, HTM APIs, layer save/load) businesses are totally separated from the backend where all the computation and memory management happens. This allows Etaler to be easily expanded and optimized. Have multiple GPUs? Just spawn multiple GPU backends! Thou shell not need any black-magic.

### Why the name?
Etaler is named after the inverse of the word "relate" for no specific reason.

## Examples

See the `examples` folder for more information. For a quick feel on how Etaler works.

Creating and printing a tensor
```C++
float arr[] = {1, 2, 3, 4};
Tensor t = createTensor(/*shape=*/{4}
                        ,/*data=*/arr);

std::cout << t << std::endl;
```

Encode a scalar
```C++
Tensor t = encoder::scalar(0.1);
```

Using the GPU
```C++
auto gpu = std::make_shared<OpenCLBackend>();
Tensor t = encoder::scalar(0.1);

//Transfer data to GPU
Tensot q = t.to(gpu);

SpatialPooler sp(/*Spatial Pooler params here*/, gpu.get());
Tensor r = sp.compute(q);
```

Saving layers
```C++
save(sp.states(), "sp.cereal");
```
## Building and platform support

| OS/Backend      |  CPU |  OpenCL                   |
|-----------------|------|---------------------------|
| Linux           |  Yes |  Yes                      |

### Dependencies
* Required
  * C++17 capable compiler
  * [Intel TBB](https://github.com/01org/tbb)
  * [cereal](https://github.com/USCiLab/cereal)

* OpenCL Backend
  * OpenCL and OpenCL C++ wrapper
  * OpenCL 1.2 capcble GPU

* Tests
  * [catch2](https://github.com/catchorg/Catch2)

### Build
Clone the repository. Then after fulfilling the dependencies. Execute `cmake` and then run whatever build system you're using.

For example, on Linux.

```shell
mkdir build
cd build
cmake ..
make -j8
```

Some cmake options are available:

| option                |  description               |  default |
|-----------------------|----------------------------|----------|
| CMAKE_BUILD_TYPE      |  Debug or Release build    |  Release |
| ETALER_ENABLE_OPENCL  |  Enable the OpenCL backend |  OFF     |
| ETALER_BUILD_EXAMPLES |  Build the examples        |  ON      |
| ETALER_BUILD_TESTS    |  Build the tests           |  ON      |

## LICENSE
Etaler is licensed under BSD 3-Clause License. So use it freely!

Be aware tha [Numenta](https://numenta.com/) holds the rights to HTM related patents. And only allows free use of their patents for non-commercial purpose. If you are using Etaler commercially; please contact Numenta for licensing. <br>
(tl;dr Etaler is free for any purpose. But HTM is not for commercial use.)

## Contribution
HTM Theory is in it's young age and as we are growing. We'd like to get contributions from you to accelerate the development of tiny-dnn! Just fork, make changes and launch a PR!

## Notes

* NVIDIA's OpenCL implementation might not report error correctly. It can execute kernels with a invalid memory object without telling you and crash a random thing the next time. If you are encountering weird behaviours. Please try [POCL](http://portablecl.org/) with the CUDA backend or use a AMD card. But the OpenCL kernels hasn't been optimized against vector processors like AMD's. You might experience performance drops doing so.

* Due to the nature of HTM. The OpenCL backend uses local memory extensively. And thus you will experience **lower than expected** performance **on processors that uses global memory to emulate local memory**. This includes but not limited to (and non of them are tested): ARM Mali GPUs, VideoCore IV GPU, any CPU.

* By default Etaler saves to a portable binary file. Etaler automatically saves as JSON when you specified a `.json` file extention. But note that JSON is fat compared to the binary format and grows fast. Make sure you know what you are doing. It is mainly for debugging.

* FPGA based OpenCL are not supported for now. FPGA platforms don't provide online (API callable) compilers that Etaler uses for code generation.

* The maxium size of SP/TM that the OpenCL backend can handle depends on how much local memory your device has. Version of OpenCL kernels that don't use local memory are planned.

* DSP/CPU/Xeon Phi based OpenCL should work out of the box. But we didn't test that.

* Etaler's Tensor implementation is more like a buffer for the time being. They are reference counted and copies when indexed(not yet implemented). (Works like xtensor's xarray)

## For NuPIC users
Etaler tho provides basically the same feature, is very different from NuPIC. Some noticeable ones are:
* Data Orientated Design instead of Object Orientated
* No Network API (planned in the future, by another repo)
* SDR is handled as a Tensor instead of a sparse matrix
* Swarming is not supported nor planned


## Things to be done before release

* [x] Implement SP
* [x] Implement Encoders
* [x] Implement TM
* [ ] Anomaly detectors
* [ ] Classifier
* [ ] Serialization
* [ ] Tests

## TODO

* [ ] C++20 Modules when C++20 is released
* [ ] OpenCL support
  * [x] Printable OpenCL Tensors
  * [x] SP in OpenCL
  * [ ] TM in OpenCL
  * [ ] Pack OpenCL kernels for distribution
* [ ] Python binding
  * [ ] Numpy inter-op via xtensor
* [ ] Windows support
* [ ] OS X support
* [x] Backend to backend data transfer
* [x] Parallel processing on CPU
* [ ] Test on ARM64
* [ ] Test on PPC64
* [ ] Load/run-able on ROOT and cling
* [ ] As VC4CL is very experimental. - Running on the RPi GPU
* [ ] Make FindEraler.cmake
* [ ] Altera AOCL support
* [x] Serialize to...
  * [x] Cereal
  * [x] JSON (via cereal)
* [ ] SP Boosting support
* [ ] Make the algorithms compliant to BAMI

## Deisgn decitions

* Don't care about x86-32 (Any 32bit architecture in fact). But no intentional breaks
* Use DOD. OOP is slow and evil
* Parallel/async-able
* Seprate the compute backend from the API frontend
* make the design scalable
* see no reason to run a layer across multiple GPUs. Just keep layers running on a single device
* Keep the API high level
* Braking the archicture is OK
* Serealizable objects should reture a `StateDict` (`std::map<std::string, std::any>`) object for serialization
  * non intrusive serialization
  * Reseves a StateDict object to deserealize
* Language binding should be in another repo
* Don't care about swarmming
* follow the KISS principle
* Configure files are evil (Looking at you NuPIC)

## Release notes


