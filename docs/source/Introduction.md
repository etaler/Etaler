# A quick introduction to Etaler

Include  Etaler's headers.

```C++
#include <Etaler/Etaler.hpp>
using namespace et;
```

Declare a HTM layer.

```C++
#include <Etaler/Algorithms/SpatialPooler.hpp>
SpatialPooler sp(/*input_shape=*/{256}, /*output_shape=*/{64});
```

Encode values

```C++
#include <Etaler/Encoders/Scalar.hpp>
Tensor x = encoders::scalar(/*value=*/0.3,
	/*min_val=*/0.f,
	/*max_val=*/5.f);
```

Run (i.e. inference) the SpatialPooler.

```c++
Tensor y = sp.compute(x);
```

If you want the SpatialPooler to learn from the data. Call the `learn()` function.

```C++
sp.learn(x, y);
```

Save the parameters of the SpatialPooler:

```C++
#include <Etaler/Encoders/Serealize.hpp>
auto states = sp.states();
save(states, "sp.cereal");

//Alternativelly save as JSON
save(states, "sp.json");
```