<div align="center">
  <img src="https://raw.githubusercontent.com/etaler/Etaler/master/docs/images/logov1.png"><br><br>
</div>

---

Etaler is a library for machine intelligence based on [HTM theory](https://numenta.com/assets/pdf/biological-and-machine-intelligence/BAMI-Complete.pdf). Providing two main features.

* HTM algorithms with modern API
* A minimal cross-platform (CPU, GPU, etc..) Tensor implementation

You can now explore HTM with modern, easy to use API and enjoy the performance boost by the GPU.

* [More about Etaler](#more-about-etaler)
* [Examples](#examples)
* [Documentation](#documentation)
* [Building and platform support](#building-and-platform-support)
  * [Dependencies](#dependencies)
  * [Building from source](#building-from-source)
* [Notes](#notes)
* [For NuPIC Users](#for-nupic-users)
* [Cite us](#cite-us)

## More about Etaler

### A GPU ready HTM library

Unlike most previous HTM implementations, Etaler is designed from the ground up to work with GPUs and allows almost seamless data transfer between CPU and GPUs.

Etaler provides HTM algorithms and a minimal Tensor implementation that operates on both CPU and GPU. You can choose what works best for you and switch between them with ease.

### Total front-end /back-end separation

Etaler is written in a way that the front-end businesses (Tensor operation calls, HTM APIs, layer save/load) are totally separated from the backend, where all the computation and memory management happens. This allows Etaler to be easily expanded and optimized. Have multiple GPUs? Just spawn multiple GPU backends! Thou shell not need any black-magic.

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

SpatialPooler sp = SpatialPooler(/*Spatial Pooler params here*/).to(gpu);
//SpatialPooler sp(/*Spatial Pooler params here*/, gpu.get()); //Alternativelly
Tensor r = sp.compute(q);
```

Saving layers

```C++
save(sp.states(), "sp.cereal");
```

## Documentation

Documents are avalible online on [Read the Docs](https://etaler.readthedocs.io/en/latest).

## Building and platform support

| OS/Backend | CPU | OpenCL |
| ---------- | --- | ------ |
| Linux      | Yes | Yes    |
| OS X       | Yes | Yes    |
| Windows    | Yes | Yes    |

* Build with GCC and libstdc++ on OS X 10.11.
* Clang should work after OS X 10.14. See [BuildOnOSX.md](docs/BuildOnOSX.md)
* Build with Visual Studio 2019 on Windows. See [BuildOnMSVC.md](docs/BuildOnMSVC.md)

### Dependencies

* Required
  
  * C++17 capable compiler
  * [Intel TBB](https://github.com/01org/tbb)
  * [cereal](https://github.com/USCiLab/cereal)

* OpenCL Backend
  
  * OpenCL and OpenCL C++ wrapper
  * OpenCL 1.2 capable GPU

* Tests
  
  * [catch2](https://github.com/catchorg/Catch2)

Notes:

1. Make sure to setup a `TBBROOT` environment variable to point to the binary installation directory of TBB. And the TBB `tbbvars.sh` file has been modified correctly and run, before running `cmake`.
2. `cereal` can be git cloned into the `Etaler/Etaler/3rdparty` directory.
3. Only the [catch.hpp](https://github.com/catchorg/Catch2/releases/download/v2.8.0/catch.hpp) file is required from Catch2, and that file can be placed into the `Etaler/tests` directory.

### Building from source

Clone the repository. Then after fulfilling the dependencies. Execute `cmake` and then run whatever build system you're using.

For example, on Linux.

```shell
mkdir build
cd build
cmake ..
make -j8
```

Some cmake options are available:

| option                             | description                                | default |
| ---------------------------------- | ------------------------------------------ | ------- |
| CMAKE_BUILD_TYPE                   | Debug or Release build                     | Release |
| ETALER_ENABLE_OPENCL               | Enable the OpenCL backend                  | OFF     |
| ETALER_BUILD_EXAMPLES              | Build the examples                         | ON      |
| ETALER_BUILD_EXAMPLE_VISUALIZER    | Build visualizer example                   | OFF     |
| ETALER_BUILD_TESTS                 | Build the tests                            | ON      |
| ETALER_BUILD_DOCS                  | Build the documents                        | OFF     |
| ETALER_ENABLE_SIMD                 | Enable SIMD for CPU backend                | OFF     |
| ETALER_NATIVE_BUILD                | Enable compiler optimize for the host CPU  | OFF     |

There are also packages available for the following distributions:

* Arch Linux: [Etaler AUR package](https://aur.archlinux.org/packages/etaler-git/)

### Building in Docker/VSC
Open the folder in VSC with remote docker extension ( ext install ms-vscode-remote.remote-containers ) - the docker image and container will start automatically. If CMake Tools extension are also installed, the building will be done automaticaly also. Otherwhise, do the regular cmake procedure inside Etaler dir.

## LICENSE

Etaler is licensed under BSD 3-Clause License. So use it freely!

Be aware that [Numenta](https://numenta.com/) holds the rights to HTM related patents. And only allows free (as "free beers" free) use of their patents for non-commercial purpose. If you are using Etaler commercially; please contact Numenta for licensing. <br>
(tl;dr Etaler is free for any purpose. But HTM is not for commercial use.)

## Contribution

HTM Theory is in it's young age and as we are growing. We'd like to get contributions from you to accelerate the development of Etaler! Just fork, make changes and launch a PR!

See [CONTRIBUTION.md](docs/source/Contribution.md)

## Notes

* NVIDIA's OpenCL implementation might not report error correctly. It can execute kernels with a invalid memory object without telling you and crash a random thing the next time. If you are encountering weird behaviors. Please try [POCL](http://portablecl.org/) with the CUDA backend or use an AMD card. However the OpenCL kernels haven't been optimized against vector processors like AMD's. They should work but you might experience performance drops doing so.

* Due to the nature of HTM. The OpenCL backend uses local memory extensively. And thus you will experience **lower than expected** performance **on processors that uses global memory to emulate local memory**. This includes but not limited to (and non of them are tested): ARM Mali GPUs, VideoCore IV GPU, any CPU.

* By default Etaler saves to a portable binary file. If you want to save your data as JSON, Etaler automatically saves as JSON when you specified a `.json` file extension. But note that JSON is fat compared to the binary format and grows fast. Make sure you know what you are doing.

* FPGA based OpenCL are not supported for now. FPGA platforms don't provide online (API callable) compilers that Etaler uses for code generation.

* DSP/CPU/Xeon Phi based OpenCL should work out of the box. But we didn't test that.

## For NuPIC users

Etaler tho provides basically the same feature, is very different from Numenta's [NuPIC](https://github.com/numenta/nupic). Some noticeable ones are:

* Data Orientated Design instead of Object Orientated
* No Network API (planned in the future, by another repo)
* SDR is handled as a Tensor instead of a sparse matrix
* Swarming is not supported nor planned

## Testing

If you have the tests builded. Run `tests/etaler_test`.

We are still thinking about weather a CI is worth the trouble. C++ projects takes too long to build on most CIs so it drags the development speed.

## Cite us

We're happy that you can use the library and are having fun. Please attribute us by linking to [etaler](https://github.com/etaler/Etaler) at [https://github.com/etaler/Etaler](https://github.com/etaler/Etaler). For scientific publications, we suggest the following BibTex citation.

```Bibtex
@misc{etaler2019,
	abstract = "Implementation of Hierarchical Temporal Memory and related algorithms in C++ and OpenCL",
	author = "An-Pang Clang",
	commit = 0226cdac1f03a642a4849ad8b9d4574ef35c943c,
	howpublished = "\url{https://github.com/etaler/Etaler}",
	journal = "GitHub repository",
	keywords = "HTM; Hierarchical Temporal Memory; Numenta; NuPIC; cortical; sparse distributed representation; SDR; anomaly; prediction; bioinspired; neuromorphic",
	publisher = "Github",
	title = "{Etaler implementation of Hierarchical Temporal Memory}",
	year = "2019"
}
```

> Note: The commit number, publication year shown above are the ones when we last update the citation. You can update the fields to match the version you uses.
