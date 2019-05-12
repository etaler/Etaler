# Using with cling/ROOT

`cling` is a JITed C++ interpter and `ROOT` is a data analyisis framework developed by CERN using cling (since version 6). Thay can be a valuable tool when using Etaler/doing experiments by providing a interactive C++ shell. This file documents how to use Etaler within clang/ROOT

## General solution
After installing Etaler to system abd launching cling or ROOT. You can load Etaler by using the `#pragma cling load` command.

By default cling only looks in `/usr/lib`. So you'll need to specisify the full path if the library is not located there.
```c++
> cling -std=c++17

****************** CLING ******************
* Type C++ code and press enter to run it *
*             Type .q to exit             *
*******************************************
[cling]$ #pragma cling load("/usr/local/lib/libEtaler.so")
[cling]$ #include <Etaler/Etaler.hpp>
```

Or you'll have to specsify an apporate search path.
```c++
> cling -std=c++17 -L /usr/local/lib
[cling]$ #pragma cling load("Etaler")
[cling]$ #include <Etaler/Etaler.hpp>
```
Otherwise, loading the library can be simplifed:
```c++
> cling -std=c++17
[cling]$ #pragma cling load("Etaler")
[cling]$ #include <Etaler/Etaler.hpp>
```

The same solution works with ROOT too! Since ROOT by default turns on C++17 by default, the C++17 flags is not needed.
```c++
> root
   ------------------------------------------------------------
  | Welcome to ROOT 6.16/00                  https://root.cern |
  |                               (c) 1995-2018, The ROOT Team |
  | Built for linuxx8664gcc on Jan 23 2019, 09:06:13           |
  | From tags/v6-16-00@v6-16-00                                |
  | Try '.help', '.demo', '.license', '.credits', '.quit'/'.q' |
   ------------------------------------------------------------

root [0] #pragma cling load("/usr/local/lib/libEtaler.so")
root [1] #include <Etaler/Etaler.hpp>
```
The same -L flag can be applyed to ROOT.

```c++
> root -L /usr/local/lib
root [0] #pragma cling load("Etaler")
root [1] #include <Etaler/Etaler.hpp>
```

## For ROOT

If oyu want load the library programmatically. You can load Etaler via `gSystem`. You'll have to specisify the full path using this method.

```c++
> root
root [0] gSystem->Load("/usr/local/lib/libEtaler.so");
root [1] #include <Etaler/Etaler.hpp>
```

## Using Etaler under an interactive C++ shell

After loading the library. You can use the library as you would normally. (And ROOT imports the `std` namespace by default.)
```c++
root [2] using namespace et;
root [3] Tensor t = ones({3,3});
root [4] cout << t << endl;
{{ 1, 1, 1},
 { 1, 1, 1},
 { 1, 1, 1}}
root [5]
```