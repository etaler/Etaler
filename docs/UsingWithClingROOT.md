# Using with cling/ROOT

[cling](https://root.cern.ch/cling) is a JITed C++ interpter and [ROOT](https://root.cern.ch/) is a data analyisis framework by CERN built upon cling (since version 6). Thay are a valuable tool that provides a interactive C++ shell when using Etaler/doing experiments. This file documents how to use Etaler within clang/ROOT.

## General solutions

After installing Etaler to system and launching cling or ROOT. You can load Etaler via the `#pragma cling load` command.

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

The same solution works with ROOT too! By default ROOT is compiled with C++11 only. You'll need to compile your own version of ROOT with C++17 enabled (by using `cmake -Dcxx17=ON`). Or you'll need to down load a version with C++17 enabled. Like the one in [Arch Linux's repo](https://www.archlinux.org/packages/community/x86_64/root/)
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

## For cling
Cling accepts the `.L` command. (ROOT also has a .L coomand, but it is for a different function)
```c++
> cling -std=c++17 -L /usr/local/lib
[cling]$ .L Etaler
[cling]$ #include <Etaler/Etaler.hpp>
```

## For ROOT

If you want load the library programmatically. You can load Etaler via `gSystem`. You'll have to specisify the full path using this method.

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
