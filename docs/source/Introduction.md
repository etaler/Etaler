With a lack of writing ablilty. The introduction will be a clone of PyTorch's introduction document. This is also a good chance to see how well Etaler is doing at it's tensor operators.

# A quick introduction to Etaler

At Etaler's core are Tensors. They are generalized matrix in more then 2 dimentions; or called N-Dimentional Arrays in some places. We'll see how they are used in-depth later. Now, let's look what we could do with tensors

```C++
// Etaler.hpp packs most of the core headers together
#include <Etaler/Etaler.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
using namespace et;

#include <iostream>
using namespace std;
```

## Creating Tensors

Tensors can be created from pointers holding the data and an appropriate shape.

```C++
int d[] = {1, 2, 3, 4, 5, 6, 7, 8};
Tensor v = Tensor({8}, d);
cout << v << endl;

// Create a matrix
Tensor m = Tensor({2, 4}, d);
cout << m << endl;

// Create a 2x2x2 tensor
Tensor t = Tensor({2, 2, 2}, d);
cout << t << endl;
```

Out

```
{ 1, 2, 3, 4, 5, 6, 7, 8}
{{ 1, 2, 3, 4}, 
 { 5, 6, 7, 8}}
{{{ 1, 2}, 
  { 3, 4}}, 

 {{ 5, 6}, 
  { 7, 8}}}
```

Just like a vector is a list of scalars, a matrix is a list of vectors. A 3D Tensor is a list of matrices. Think it like so: indexing into a 3D Tensor gives you a matrix, indexing into a matrix gives you a vector and indexing. into a vector gives you a scalar,

Just for clearifcation. When I say "Tensor" I ment a `et::Tensor` object. 

```C++
// Index into v wii give you a scalar
cout << v[{0}] << endl;

// Scalars are special as you can convert them into native C++ types
cout << v[{0}].item<int>() << endl;

// Indexing a matrix gives you a vector
cout << m[{0}] << endl;

// And indexing into a 3d Tensor to get a matrix
cout << t[{0}] << endl;
```

Out

```
{ 1}
1
{ 1, 2, 3, 4}
{{ 1, 2}, 
 { 3, 4}}
```

You can also create tensors of other types. The default, as you can see, is whatever the pointer is point to. To create floating-point Tensors, just point to an floating point array then call `Tensor()`.

You can also create a tensor of zeros with an supplied shape using `zeros()`

```C++
Tensor x = zeros({3, 4, 5}, DType::Float);
cout << x << endl;
```

Out

```
{{{ 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}}, 

 {{ 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}}, 

 {{ 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}, 
  { 0, 0, 0, 0, 0}}}
```

## Operation with Tensors

You can operate on tensors in the ways you would expect.

```C++
Tensor x = ones({3});
Tensor y = constant({3}, 7);
Tensor z = x + y;
cout << z << endl;
```

Out 

```
{ 8, 8, 8}
```

One helpful operation that we will make use of later is concatenation.

```C++
// By default, the cat/concat/concatenate function works on the 1st axis
Tensor x_1 = zeros({2, 5});
Tensor y_1 = zeros({3, 5});
Tensor z_1 = cat({x_1, y_1});
cout << z_1 << endl;

// Concatenate columns:
x_2 = zeros({2, 3});
y_2 = zeros({2, 5});
z_2 = cat({x_2, y_2}, 1);
cout << z_2 << endl;
```

Out

```
{{ 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0}}
{{ 0, 0, 0, 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0, 0, 0, 0}}
```

## Reshaping Tensors
Use the `reshape` method to reshape a tensor. Unlike PyTorch using `view()` to reshapce tensots. view in Etaler works like the one in [xtensor](https://github.com/xtensor-stack/xtensor); it performs indexing.


```C++
Tensor x = zeros({2, 3, 4});
cout << x << endl;
cout << x.reshape({2, 12}) << endl;  // Reshape to 2 rows, 12 columns
```

Out

```
{{{ 0, 0, 0, 0}, 
  { 0, 0, 0, 0}, 
  { 0, 0, 0, 0}}, 

 {{ 0, 0, 0, 0}, 
  { 0, 0, 0, 0}, 
  { 0, 0, 0, 0}}}
{{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
```

## HTM Algorithms

HTM algorithms, like what the name indicates. Implements HTM related algorithms. They are the sole perpose why Etaler exists.

Without getting too deep. Typically the first thing we do in HTM is encode values into SDRs. SDRs are sparse binary tensors. i.e. Elements in the tensor are either 1 or 0 and most of them are 0s.

```C++
Tensor x = encoder::scalar(/*value=*/0.3,
	/*min_val=*/0.f,
	/*max_val=*/5.f);
cout << x << endl;
```

Out

```
{ 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
```

[Spatial Pooler](https://numenta.com/neuroscience-research/research-publications/papers/htm-spatial-pooler-neocortical-algorithm-for-online-sparse-distributed-coding/) is a very commonly used layer in HTM. Trough an unsurpervised learning process, it can extract patterns from SDRs fed to it.


```C++
SpatialPooler sp(/*input_shape=*/{32}, /*output_shape=*/{64});
// Run (i.e. inference) the SpatialPooler.
Tensor y = sp.compute(x);
cout << y.cast(DType::Bool) << endl;
// If you want the SpatialPooler to learn from the data. Call the `learn()` function.
sp.learn(x, y);
```

Out
```
{ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
```

After traning, you might want like to save the Spatial Pooler's weights for use in the future.

```C++
auto states = sp.states();
save(states, "sp.cereal");

//Alternativelly save as JSON
save(states, "sp.json");
```