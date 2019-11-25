# Developer Notes

This file documents some facts about the codebase that might be useful for it's developers and users.

* The `Error.hpp` file provides the `et_assert()` macro for function pre/post condition checking. It should be replaced with C++20 contracts when avaliable.
  * Use it to check for function pre and post condition fails.
  * You can also throw an `EtError` if you like it for recoverable errors.
  * Not disabled in release builds

* `Error.hpp` also provides the `ASSERT()` macro that works exactly like C `assert()`. But
  * Does not generate a unused variable warning in a release build

* Address Sanitizer is your good friend to debug buffer overflows if you are on Linux or OS X.
  * But it might cause Etaler not able to detect OpenCL platforms.
  * Memory Sanitizer will work with OpenCL. But is too sensitive.

* Nvidia's OpenCL implementation although very optimized, has piles upon piles of problems.
  * Use POCL w/ CUDA backend for debugging. POCL is a lot slower, but very stable.
  * Or use Intel/AMD's OpenCL SDK
