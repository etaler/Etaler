# Building with Visual Studio 2019

This is a step by step instruction detailing a proven way to build Etaler with Visual Studio 2019.

## Install dependencies

- Install [CMake](https://cmake.org)
- Install [Git Bash](https://gitforwindows.org)
- Using a Git Bash terminal, perform the following commands:
``` shell
cd Etaler/3rdparty
git clone https://github.com/imneme/pcg-cpp.git
git clone https://github.com/USCiLab/cereal.git
```

- Download the latest Intel TBB zip file.
- Extract it into `C:\Program Files\Intel\TBB`
- Update PATH to include `C:\Program Files\Intel\TBB\bin\intel64\vc14`
- Setup `TBBROOT` environment variable to point to `C:\Program Files\Intel\TBB`

To compile the tests you need to:
- Download standalone [catch.hpp](https://github.com/catchorg/Catch2/releases/download/v2.8.0/catch.hpp) into the `Etaler/tests` directory.

If you intend to use Etaler with a GPU:
- Install an OpenCL SDK appropriate for your GPU (e.g. NVidia CUDA SDK).
- Download Khronos [cl.hpp](https://www.khronos.org/registry/OpenCL/api/2.1/cl.hpp) into `Etaler/3rdparty/CL` directory.

## CMake

The CMake application is used to generate the project and solution files for MSVC.

Set the "Where is the source code:" text box to point to the root of your cloned Etaler repository. And set the "Where to build the binaries:" text box to point to a `build` subdirectory off of your Etaler repositories.

For example:
- Source: `C:/Github/Etaler`
- Binaries: `C:/Github/Etaler/build`

Press the `Configure` button and choose the "Visual Studio 16 2019" generator. Finally press the `Generate` button.

Visual Studio 2019 can then be used to load the `Etaler/build/Project.sln` solution file, and build the library, tests and examples.

By default the solution builds a **dynamic** linked Etaler library. Therefore, the `Etaler/build/Etaler/Release` (or `Debug`) directory needs to be in your `PATH`. Or the `Configuration Properties -> Debugging -> Environment` entry can use "PATH=%PATH%;C:/Github/Etaler/build/Etaler/Release".
