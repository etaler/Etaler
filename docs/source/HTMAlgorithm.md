# Introduction to HTM Algorithms

Hierarchical Temporal Memory is a biologically possible machine learning algorithm developed by [Numenta](https://numenta.com/). Biological possiblity means the algorithms has to simulate (or approximate) behaivour and limitations of real neurons. i.e. Neurons has to speak binary ([spikes](https://en.wikipedia.org/wiki/Biological_neuron_model)) and there's no global learning (no [backpropergation](https://en.wikipedia.org/wiki/Backpropagation)). In HTM Theory, this means HTM operates on [Sparse Distributed Representation](https://en.wikipedia.org/wiki/Sparse_distributed_memory) (SDR, basically binary tensors) and every layer operates indipendently.

This document will introduce HTM algorithms as a black box. It is highly reommended to watch the [HTM School](https://www.youtube.com/watch?v=XMB0ri4qgwc&list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9) series on YouTube for deeper understanding of the material.

## Encoders

Encoders are how data interact with HTM. As mentioned earlier, HTM can only accepts binary as input and only generates binary output. So a method of converting data into SDRs is required. There are several encoders avaliable by default.

| Encoder    | Usage                           | 
|------------|---------------------------------| 
| scalar     | scalar of a defined range       | 
| category   | encodes category                | 
| gridCell1d | arbitrary scalar                | 
| gridCell2d | arbitrary 2D vector             | 

All encoders (for now) are plan function calls in the `et::encoder` namespace. And every of them returns a tensor of Booleans; you can inspect the result of any encoder by printing the returned value.

```C++
auto sdr = encoder::gridCell1d(0.5);
cout << sdr << endl;
```

Out

```
{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
 0, 0, 0, 0, 0, 0}
```

The rule of thumb for an encoders is that for any given pair of value encoded. The farther the values are, the less 1 bits in the resulting SDR overlaps. The short [ROOT](https://root.cern.ch) script demonstrates this property (for a [grid cell](https://en.wikipedia.org/wiki/Grid_cell) encoder).

```C++
root [] g = TGraph(100);
root [] x = encoder::gridCell1d(0.5);
root [] for(int i=0;i<100;i++) {
root []     float p = (float)i/100;
root []     auto s = encoder::gridCell1d(p);
root []     int overlap = sum(x && s).item<int>();
root []     g.SetPoint(i+1, p, overlap);
root [] }
root [] g.Draw();
root [] g.SetTitle("# of overlapping bits");
```

Out

![number_of_overlapping_bits](../images/gc1d_overlap.svg)

### Extending encodings

Unless in a very simple case, you might have multiple fields or non-standard types in your data. In such case, you might need to come up with your own encoder or use multiple of them together. Having your own encoder is easy - the only requirment of an encoder is **a)** returns a fixed length binary tensor **b)** the tensor have to be sparse and **c)** values futher apart should have less and less 1 bits overlapping. **a and b** are requirments. Failing to fulill will lead to a program crash. While **c** is nice to have but not having it will only causing bad learning results or not learning at all.

For example. We can come up with an basic scalar encoder that encodes values from 0~1.

```C++
Tensor dumb_encoder(float v)
{
    intmax_t start = v*(64-4);
    // Create a tensor of length 64.
    // Then set 4 of the bits to 1
    Tensor sdr = zeros({64}, DType::Bool);
    sdr[{range(start, start+4)}] = 1;
    return sdr;
}

cout << dumb_encoder(0.1) << endl;
```

Out

```
{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
```

It is quite hard to build a robust encoder for complex data and in most cases the data are not related to each other. So pratically we would apply encoding to each field then concatenate all SDRs together.

```C++
Tensor complex_encoder(float v1, float v2, float v3)
{
        Tensor sdr1 = encoder::scalar(v1);
        Tensor sdr2 = encoder::scalar(v2);
        Tensor sdr3 = encoder::scalar(v3);
        return concat({sdr1, sdr2, sdr3});
}

cout << complex_encoder(0.1, 0.2, 0.3) << endl;
```

Out

```
{ 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 
 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
```

## Spatial Pooler

[Spatial Pooler](https://numenta.com/neuroscience-research/research-publications/papers/htm-spatial-pooler-neocortical-algorithm-for-online-sparse-distributed-coding/) is a dimention reduction/clustering algorithm. As a black box, a Spatial Pooler learns the associations between bits (learning which groups of bits frequently becomes a 1 together) and merges them together. The process removes reducdent information within the SDR and enforces a stable density (number of 1 bits in the SDR). Improving Temporal Memory's performance.

```C++
auto sp = SpatialPooler(/*input_shape=*/{256}, /*output_shape=*/{64});
auto y = encoder::gridCell1d(0.5);
cout << sp.compute(y) << endl;
```

Out

```
{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1}
```

An untrained Spatial Pooler is pretty much useless; knowing nothing about the input SDR and only guess the relation among bits. We need to feed data to it and allow it to learn the relations itself - by feeding the results back into the `learn` method.

```C++
auto sp = SpatialPooler(/*input_shape=*/{256}, /*output_shape=*/{64});
for(const auto& sample : dataset) {
    auto x = encode_data(sample);
    auto y = sp.compute(x);

    sp.learn(x, y); // Ask the SP the learn!
}
```

Like any machine learning algorithm, the Spatial Pooler have multiple hyper-parameters. The important ones include:

```C++
sp.setGlobalDensity(density); // The density of the generated SDR
sp.setBoostingFactor(factor); // Allows under-performing cells to activate
sp.setActiveThreshold(thr);   // How much activity can lead to cell activation

sp.setPermanenceInc(inc);     // These are the learning rates
sp.setPermanenceDec(dec);     // For both reward and punish
```

## Temporal Memory

As the name implied, [Temporal Memory](https://numenta.com/neuroscience-research/research-publications/papers/why-neurons-have-thousands-of-synapses-theory-of-sequence-memory-in-neocortex/) is a sequence memory. It learns the relations of bits at time `t` and `t+1`. For a high level view, given a Temporal Memory layer is trained on the sequence A-B-C-D. Then asking what is after A, the TM layer will respond B.

```C++
// Some setup is requied for a TM
// In this example, the input shape is {256} and there are 16 cells per column
// last_active and last_pred are the stats of the TM
// This is annoning and will be fixed in a future release
Tensor last_active = zeros({256, 16}, DType::Bool);
Tensor last_pred = zeros({256, 16}, DType::Bool);

auto tm = TemporalMemory(/*input_shape*/{256}, /*cells_per_column=*/16);
auto [pred, active] = tm.compute(encoder::gridCell1d(0.5), last_pred);
cout << pred << endl;
```

Out

```
{{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
 ....
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
```

The Temporal Memory has some very good properties. First, a Temporal Memory will respond "nothing" when it has no idea what is going on (unlike neural networks, they return at least something when met with unknown). Like in this case, we asked a Temporal Memory that has been trained on nothing to predict what's after 0.5. The layer responded with a zero tensor of shape `{input_shape, cells_per_column}`. Indicating nothing.

In most applications. We don't care about what's happning within each column but weather a column contains a cell that is a 1. This can be extracted with a `sum()` call.

```C++
// Continuing from the previous block
auto pred_sdr = sum(pred, 1, DType::Bool);
cout << pred_sdr << endl;
```

Out

```
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ...
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0}
```

Traning a Temporal Memory involved basically the same process as the Spatial Pooler. But with slightly more steps.

```C++
// Continuing from the previous block
for(const auto& sample : dataset) {
    auto x = encode_data(sample);
    auto [pred, active] = tm.compute(x, last_active);
    // You can you the predictions here
    ...
    // Now update the states
    tm.learn(active, last_state);
    tie(last_pred, last_active) = pair(pred, active);
}
```

Idealy, unless your input data is very simple. We recommend to pass the encoded through a SpatialPooler to reduce useless information.

```C++
for(...) {
    auto x = encode_data(...);
    auto y = sp.compute(x); // Use the SP!

    auto [pred, active] = tm.compute(y, last_active);
    ...
}
```

After traning, the TM should be able to predict what is possible in the next time step based on current input and the state. A Temporal Memory layer can gracefully deal with ambiguous situaction. When trained on the sequence A-B-C-B-C-D then asking what's after C without a context(past state), the TM will respond both B and D.

### Detection anomaly

One of HTM's main use is to perform anomaly detection. The method is stright forward. Given a well trained Spatial Pooler, Temporal Memory and a cyclic signal. The only cause for the TM to not predicting well must be an anomaly in the signal. The TM's property ties in very well with the application. A TM will resolve ambiguous states by predicting everything and predicts nothing when it don't know.

We can compute the anomaly score using the TM's prediction from the last time step and the actual input from the current time step.

```C++
for(...) {
    auto x = encode_data(...);
    auto [pred, active] = tm.compute(y, last_active);

    float anomaly_score = anomaly(sum(last_pred, 1, DType::Bool), x);
    ...
}
```

Do note that detecting anomaly at `t=0` will always give you an anomaly score of 1 due to having no past history to rely on.

## Classifers

It's good to have the ability to predict the future. But sometimes it's hard to make sense of the predictios in the SDR form. Etaler provides a biologically possible classifer to categorize results back into human readable information.

**Note:** The API of classifers are pron to change. And more classifers will be added.

```C++
auto clf = SDRClassifer(/*input_shape=*/{256}, /*num_class=*/10);

//Add SDRs to the classifer
for(int i=0;i<10;i++)
    clf.addPattern(encoder::gridCell1d(i/10.f), i);

cout << "The value 0.8 is close to "
    << clf.compute(encoder::gridCell1d(0.8))/10.f << endl;
```

Out

```
The value 0.8 is close to 0.8
```