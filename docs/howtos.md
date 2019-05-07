# How-Tos

Some examples about how Etaler's API works


## Tensor
Tensors are how Etaler stores data. They are a minimal NDArray implementation. Thus it is currently lacking some common features. But they should be enough for HTM.

### Create
It is easy to create a Tensor
```C++
Tensor t = Tensor(/*shape=*/{4,4}, DType::Float);
```
At this point, the contents in the Tensor hasn't been initalized. To initalize call `createTensor` with a pointer pointing to a plan array containing the data you want the Tensor to have. The Tensor object will copy all of the provided data to it's internal memory.

```C++
float data[4] = {3,2,1,6};
Tensor t = Tensor(/*shape=*/{4}, DType::Float, data);
```

### Copy
Tensors holds a `shared_ptr` object that points to a acuatl implementation provided by the backend. Thus, copying via `operator =` and the copy constructor results in a shallow copy.

```C++
Tensor t = Tensor({4,4});
Tensor q = t; //q and t points to the same internal object
```

Use the `copy()` member function to perform a deep copy.
```C++
Tensor t = Tensor({4,4});
Tensor q = t.copy();
```

### Accessing the raw data held by the Tensor
If the internal implementaion allows. You can get a raw pointer pointing to where the Tensor stores it's data. Otherwise a `nullptr` is returned.

```C++
Tensor t = Tensor({4,4}, DType::Int32);
int32_t* ptr = (int32_t*)t.data();
```

### Copy data from Tensor to a std::vector
Nomatter where or how a Tensor stores it's data. the `toHost()` method copies everything into a std::vector.

```C++
Tensor t = Tensor({4,4}, DType::Int32);
std::vector<int32_t> res = t.toHost<int32_t>();
```

**Be aware** that due to std::vector<bool> is a specialization for space and the internal data cannot be accessed easily, use uint8_t instead.
```C++
Tensor t = Tensor({4,4}, DType::Bool);
std::vector<uint8_t> res = t.toHost<uint8_t>();
//std::vector<bool> res = t.toHost<bool>();//This will fail
```

Also that `toHost` checks that the type you are stroing to is the same that the Tensor holds. If they do not match, an exception is thrown.
```C++
Tensor t = Tensor({4,4}, DType::Float);
std::vector<int32_t> res = t.toHost<int32_t>(); //Throws
```

### Indexing
Etaler supports basic indexing using Torch's syntax
```
Tensor t = ones({4,4});
Tensor q = t.view({2, 2}); //A vew to the value of what is at position 2,2
std::cout << q << std::endl; // Prints {1}
```

Also ranged indexing
```
Tensor t = ones({4,4});
Tensor q = t.view({range(2), range(2)}); //A vew to the value of what is at position 2,2
std::cout << q << std::endl;
// Prints
// {{1, 1},
    {1, 1}}
```

### Add, subtract, multiply, etc...
Not supported.

### Brodcasting
Not supported.

### Copy Tensor from backend to backend
If you have multiple backends (ex: one on the CPU and one for GPU), you can easily transfer data between the backends.
```C++
auto gpu = make_shared<OpenCLBackend>();
Tensor t = createTensor({4,4}, DType::Float);
Tensor q = t.to(gpu);
```