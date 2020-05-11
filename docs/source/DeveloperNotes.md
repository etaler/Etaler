# Developer Notes

This file documents some facts about the codebase that might be useful for it's developers and users.

* `Error.hpp` provides a few ways to check if the program is runnign correctly.
* `et_assert` works like C asserts. But it is not dieabled in a release build.
  * Aborts the entire program when condition check fails.
  * The least overhead
  * Use it in core/lowlevel code
* `et_check` throws an `EtError` exception.
  * More overhead
  * Recoverable
* `ASSERT` is C assert but doesn't cause unused variable warning.
* `et::enableTraceOnException(bool)` sets if a stack trace is generated while creating a `EtError` exception
   * On by default. Can be disabled in runtime in production.
   * Very useful in debugging and interactive coding
   * Stack trace appends to the error message

* Address Sanitizer is your good friend to debug buffer overflows if you are on Linux or OS X.
  * But it might cause Etaler not able to detect OpenCL platforms.
  * Memory Sanitizer will work with OpenCL. But is too sensitive.

* Nvidia's OpenCL implementation although very optimized, has piles upon piles of problems.
  * Use POCL w/ CUDA backend for debugging. POCL is a lot slower, but very stable.
  * Or use Intel/AMD's OpenCL SDK
