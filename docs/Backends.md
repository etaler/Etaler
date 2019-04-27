# Backends

Backends are how Etaler supports computing on different device/processors. They perform the acuatl computing and memory managment. Currently there are 2 backends avaliable.

* CPUBackend
* OpenCLBackend

They should cover more than 99% of the use cases. When needed, other backends can be added to support new devices.

## Front/backend speration


## Backend name
You can get the backend's name usging the `name()` method.
```C++
std::cout << backend->name() << "\n";//prints: CPU
```

## Default Backend
Etaler provices a convient function `defaultBackend()` that returns a default/usable backend for convience.
When no backends are provied to algorithms/objects the default backend is used.

By default `defaultBackend()` will return a `CPUBackend`. You can change this by calling the `setDefaultBackend()` function.

Like so:
```C++
auto gpu = std::make_shared<OpenCLBackend>(); //Make sure this pointer is alive until exit
setDefaultBackend(gpu);
```