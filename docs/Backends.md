# Backends

Backends are how Etaler supports computing on different device/processors. They perform the actual computing and memory managment. Currently there are 2 backends avaliable.

* CPUBackend
* OpenCLBackend

They should cover more than 99% of the use cases. When needed, other backends can be added to support new devices.

## Total front-end/back-end seperation

Etaler seperated it's frontends and backends. i.e. Calling `Tensor::Tensor()` does not immidatelly create a Tensor object. Instead it calls a backend to create the object for you. Depending on which backend is called. The resulting Tensor can live on CPU's main memory or the GPU's VRAM. (Omiting a lot of details.)

This applies to all Tensor operations and HTM algorithms. All of them are provided by a backend and the Tensor object only calculates what backend-indipendent parameters are needed to generate correct result. Allowing a very simple way to have a Tensor running anywhere.

## Backend APIs

By design (might change in the future). All backend-exposed compute API are expecting thair own Tensors being passed in . _View of tensors are not expected_ (besides `realize()` and `assign()`. They handle views). If anython besides an actual Tensor is passed in, the **backend aborts**.

The frontend APIs of common functions like `sum`, `cast` automatically handles realizing. But HTM specific APIs don't.

## Backend name

You can get the backend's name usging the `name()` method.

```C++
std::cout << backend->name() << "\n";//prints: CPU
```

## Default Backend

Etaler provices a convient function `defaultBackend()` that returns a usable backend.
When no backends are provied to algorithms/objects the default backend is used.

By default `defaultBackend()` returns a `CPUBackend`. You can which backend is used by calling the `setDefaultBackend()` function.

Like so:

```C++
auto gpu = std::make_shared<OpenCLBackend>(); //Make sure this pointer is alive until exit
setDefaultBackend(gpu);
```

## Using multiple backends

Using multiple backends should be easy. Just initalize multiple backends and tensors on them! You can even have different threads controlling different backends for maxium performance. The backends are not thread-safe tho. You'll have to handle that yourself.
