# Building on OS X

This is a step by step instruction detailing on a proven way to build Etaler on OS X

## Install dependencies

``` shell
brew install tbb cereal cmake
```

## Install catch2

If you want to runs tests

```shell
git clone https://github.com/catchorg/catch2
cd catch2
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

## Get the OpenCL C++ wrapper

If you want OpenCL support. Download and install the offical OpenCL C++ wrapper

```shell
sudo wget https://www.khronos.org/registry/OpenCL/api/2.1/cl.hpp -P /System/Library/Frameworks/OpenCL.framework/Headers/
```

## Build Etaler

```shell
git clone https://github.com/Etaler/Etaler
cd Etaler
mkdir build && cd build
cmake ..
make -j4
```

Please add `-DETALER_BUILD_TESTS=off` to cmake if you don't want to build the tests or don't have catch2 installed.<br>
Add `-DETALER_ENABLE_OPENCL=on` for OpenCL support.

## Older OS X sysetms

If you are running OS X < 10.14. Please install `gcc` from homebrew and add `-DCMAKE_CXX_COMPILER=gcc-9` (Or whatever the GCC version you installed) to build Etaler. Apple doesn't ship the full C++17 features Etaler needs on the older systems.





