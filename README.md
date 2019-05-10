
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
Tensor t = Tensor(/*shape=*/{4}
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
| OS X            |  Yes |  Yes                      |

* Build with GCC and libstdc++ on OS X 10.11. Clang should work after OS X 10.14. See [BuildOnOSX.md](docs/BuildOnOSX.md)

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

Be aware tha [Numenta](https://numenta.com/) holds the rights to HTM related patents. And only allows free (as "free beers" free) use of their patents for non-commercial purpose. If you are using Etaler commercially; please contact Numenta for licensing. <br>
(tl;dr Etaler is free for any purpose. But HTM is not for commercial use.)

## Contribution
HTM Theory is in it's young age and as we are growing. We'd like to get contributions from you to accelerate the development of Etaler! Just fork, make changes and launch a PR!

See [CONTRIBUTION.md](docs/Contribution.md)

## Notes

* NVIDIA's OpenCL implementation might not report error correctly. It can execute kernels with a invalid memory object without telling you and crash a random thing the next time. If you are encountering weird behaviours. Please try [POCL](http://portablecl.org/) with the CUDA backend or use a AMD card. However the OpenCL kernels haven't been optimized against vector processors like AMD's. They should work but you might experience performance drops doing so.

* Due to the nature of HTM. The OpenCL backend uses local memory extensively. And thus you will experience **lower than expected** performance **on processors that uses global memory to emulate local memory**. This includes but not limited to (and non of them are tested): ARM Mali GPUs, VideoCore IV GPU, any CPU.

* By default Etaler saves to a portable binary file. If yout want to save your data as JSON, Etaler automatically saves as JSON when you specified a `.json` file extention. But note that JSON is fat compared to the binary format and grows fast. Make sure you know what you are doing.

* FPGA based OpenCL are not supported for now. FPGA platforms don't provide online (API callable) compilers that Etaler uses for code generation.

* DSP/CPU/Xeon Phi based OpenCL should work out of the box. But we didn't test that.

## For NuPIC users
Etaler tho provides basically the same feature, is very different from NuPIC. Some noticeable ones are:
* Data Orientated Design instead of Object Orientated
* No Network API (planned in the future, by another repo)
* SDR is handled as a Tensor instead of a sparse matrix
* Swarming is not supported nor planned

## Testing
If you have the tests builded. Run `tests/etaler_test -s`.

We are still thinking about weather a CI is worth the problem. C++ projects takes too long to build on most CIs and is a problem for fast development.

## Things to be done before release

* [x] Implement SP
* [x] Implement Encoders
* [x] Implement TM
* [x] Anomaly detector
* [ ] Classifier
* [ ] Serialization
* [x] Tests

## TODO

* [ ] C++20 Modules when C++20 is released
* [ ] OpenCL support
  * [x] Printable OpenCL Tensors
  * [x] SP in OpenCL
  * [x] TM in OpenCL
  * [ ] Pack OpenCL kernels for distribution
* [ ] Python binding
  * [ ] Numpy inter-op via xtensor
* [ ] Windows support
* [x] OS X support
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
* [x] Basic Tensor indexing

## Release notes
