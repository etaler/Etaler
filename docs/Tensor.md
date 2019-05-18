# Tensor
Tensors are how Etaler stores data. They are a minimal NDArray implementation. Thus it is currently lacking some features. But they should be enough for HTM.

For now content type of `int`, `bool` and `float` are supported.

## Creating a Tensor

It is easy to create a Tensor

```C++
Tensor t = Tensor(/*shape=*/{4,4}, DType::Float);
```

At this point, the contents in the Tensor hasn't been initalized. To initalize, call `Tensor()` with a pointer pointing to a plan array. The Tensor object will copy all of the provided data and use them for initalization.

```C++
float data[4] = {3,2,1,6};
Tensor t = Tensor(/*shape=*/{4}, DType::Float, data);
```

## Copy
Tensors holds a `shared_ptr` object that points to a actual implementation provided by the backend. Thus, copying via `operator =` and the copy constructor results in a shallow copy.

```C++
Tensor t = Tensor({4,4});
Tensor q = t; //q and t points to the same internal object
```

Use the `copy()` member function to perform a deep copy.
```C++
Tensor t = Tensor({4,4});
Tensor q = t.copy();
```

## Accessing the raw data held by the Tensor
If the implementaion allows. You can get a raw pointer pointing to where the Tensor stores it's data. Otherwise a `nullptr` is returned.

```C++
Tensor t = Tensor({4,4}, DType::Int32);
int32_t* ptr = (int32_t*)t.data();
```

## Copy data from Tensor to a std::vector
No matter where or how a Tensor stores it's data. the `toHost()` method copies everything into a std::vector.

```C++
Tensor t = Tensor({4,4}, DType::Int32);
std::vector<int32_t> res = t.toHost<int32_t>();
```

**Be aware** that due to `std::vector<bool>` is a specialization for space and the internal data cannot be accessed by a pointer, use uint8_t instead.
```C++
Tensor t = Tensor({4,4}, DType::Bool);
std::vector<uint8_t> res = t.toHost<uint8_t>();
//std::vector<bool> res = t.toHost<bool>();//This will not compile
```

Also `toHost` checks that the type you are stroing to is the same the Tensor holds. If mismatch, an exception is thrown.
```C++
Tensor t = Tensor({4,4}, DType::Float);
std::vector<int32_t> res = t.toHost<int32_t>(); //Throws
```

## Indexing
Etaler supports basic indexing using Torch's syntax
```C++
Tensor t = ones({4,4});
Tensor q = t.view({2, 2}); //A vew to the value of what is at position 2,2
std::cout << q << std::endl; // Prints {1}
```

Also ranged indexing
```C++
Tensor t = ones({4,4});
Tensor q = t.view({range(2), range(2)}); //A view to the value of what is at position 2,2
std::cout << q << std::endl;
// Prints
// {{1, 1},
//  {1, 1}}
```

The `all()` function allows you to specsify the entire axis.
```C++
Tensor t = ones({4,4});
Tensor q = t.view({all(), all()});//The entire 4x4 matrix
```

## Writing data trough a view

And you can write back to the source Tensor using `assign()`
```C++
Tensor t = ones({4,4});
Tensor q = t.view({2,2});
q.assign(zeros({2,2}));
std::cout << t << '\n';
//Prints
// {{ 0, 0, 1, 1},
//  { 0, 0, 1, 1},
//  { 1, 1, 1, 1},
//  { 1, 1, 1, 1}}
```

The Python style method works too tanks to C++ magic
```C++
Tensor t = ones({4,4});
t.view({2,2}) = zeros({2,2});
std::cout << t << '\n';
//Prints
// {{ 0, 0, 1, 1},
//  { 0, 0, 1, 1},
//  { 1, 1, 1, 1},
//  { 1, 1, 1, 1}}
```

But assigning to an instance of view doesn't work. Jusk like how things are in Python.
```C++
Tensor t = ones({4,4});
Tensor q = t.view({2,2});
q = ones({2,2});
std::cout << t << '\n';
//Prints
// {{ 1, 1, 1, 1},
//  { 1, 1, 1, 1},
//  { 1, 1, 1, 1},
//  { 1, 1, 1, 1}}
```

### Technical note
Numpy uses the `__set_key__` operator to determine when to write data. If the operator is not called. Python itself handles object reference assignment and thus data is not written.
However thers is no such  mechanism  in C++. So Etaler distingishs when to copy the reference it holds and when to write data using `operatpr= ()&` and `operator= ()&&`. When writing to an l-valued Tensor, the reference is copied. While assigning to an r-value, actual data is copied trough the view.

Which works in most cases, but there are caveats.
```C++
Tensor foo(const Tensor& x)
{
        return x;
}

foo(x) = ones({...}); //Oops. Data is written to x even tho it is passed as const!
```

## Add, subtract, multiply, etc...
Common Tensor operations are supported. Incluing +, -, *, /, exp, log(ln), negation, Tensor comparsions, and more! Use them like how you would in Python. The comparsion operators alowys return a Tensor to bools. The others return a Tensor of what you get in plan C/C++ code.

```
Tensor a = ones({4,4});
Tensor b = a + a;
```

## Brodcasting
Etaler supports PyTorch's brodcasting rules without the legacy rules. Any pair of Tensors are bordcastable if the following rules holds true.

(Stolen from PyTorch's document.)
* Each tensor has at least one dimension.
* When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

For example:
```C++
Tensor a, b;

//The trailing dimensions match
a = ones({2, 2});
b = ones({   2});
std::cout << (a+b).shape() << std::endl;
// {2, 2}

//But not necessary
a = ones({6, 4});
b = ones({1});
std::cout << (a+b).shape() << std::endl;
// {6, 4}

//This fails
a = ones({2, 3});
b = ones({   2});
std::cout << (a+b).shape() << std::endl;
//Fails
```

## Copy Tensor from backend to backend
If you have multiple backends (ex: one on the CPU and one for GPU), you can easily transfer data between the backends.
```C++
auto gpu = make_shared<OpenCLBackend>();
Tensor t = zeros({4,4}, DType::Float);
Tensor q = t.to(gpu);
```