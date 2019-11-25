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
setDefaultBackend(std::make_shared<OpenCLBackend>());
```

## Using multiple backends

Using multiple backends should be easy. Just initalize multiple backends and tensors on them! You can even have different threads controlling different backends for maxium performance. The backends are not thread-safe tho. You'll have to handle that yourself.

## Tensors and views, how do they work

From a technical point. Each backend implements it own XXXBuffer (ex. CPUBuffer) class, storing whatever is needed. When the backend being requested to create a tensor (the `createTensor` method called). The backend returns a shared_ptr pointing to XXXBuffer, which is then wrapped by a TensorImpl. When the reference counter drops to 0, the `releaseTensor` method is called automatically (Also all TensorImpl holds a shared_ptr to the backend, so you don't need to worry about the backend being destructed before all tensors being destructed).

When creating a view. Like Numpy and PyTorch's implementation we modifies the offset and stride of the tensor.

But not all backend APIs support handling strides. (Espcally HTM algorithms and those modifies data in-place). If a strided Tensor is sent to a API that doesn't support strides. Backend aborts.