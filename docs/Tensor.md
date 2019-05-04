# Tensors

Tensor is the primary way how Etaler handles bulks of data. They are N-dimentioal arrays of a single type.

Currently Etaler's Tensor can store 3 different types of data. `float`, `int`, and `bool`.

## Creating a Tensor
Tensors can be either created un-initalized or initalized. The initial value of an un-initalized tensor is not defined but is faster to create.

To create an un-initalized tensor:
```C++
Tensor t = Tensor(/*shape=*/{4,4}
	, /*dtype=*/DType::Int32)
```

If you need deterministec initial values. You can supply a pointer to an array that stores them. Then Etaler infers the intended type and copy the given array as initial values.

```C++
int a[] = {0,1,2,3};
Tensor t = Tensor(/*shape=*/{4}
	,/*data=*/a);
```

If all you need is a Tensor of a constant. Use `constant`, or the common `zeros`, `ones` to initalize your Tensor to 0 and 1

```C++
Tensor t = constant({4,4}, 50);
//t is a 4x4 Tensor where all of it's elements are initalized to 50

Tensor q = zeros({4,4});
//q is a 4x4 Tensor of 0s
```

## Printing a Tensor
Tensors can be printed via an `std::ostream`. Thus you can print the content of a tensor via `cout`.
```C++
Tensor t = some_tensor();
std::cout << t << std::endl;
```

## Indexing and views
Etaler supports some basic indexing and unlike some C++ libraries. The indexing are lazily evaluated at runtime.

```C++
Tensor t = ones({4});
Tensor q = t.view({2});
std::cout << q << std::endl;
///Prints: {1}
```

And you can write data through a view and see the values pop back up in the original Tensor.

```C++
Tensor t = ones({4});
Tensor q = t.view({2});
q.assign(zeros({1}))
std::cout << t << std::endl;
///Prints: {1,1,0,1}
```