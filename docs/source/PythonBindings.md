# Python bindings

Currently there are no offical python support. The feature is planned. But nevertheless, you can use Etaler in Python via [ROOT](https://root.cern.ch) and it's automatic binding generation feature.

## Example

```Python
# Load ROOT
import ROOT
gROOT = ROOT.gROOT
gSystem = ROOT.gSystem

# Include Etaler headers
gROOT.ProcessLine("""
#include <Etaler/Etaler.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Algorithms/Anomaly.hpp>
#include <Etaler/Encoders/GridCell1d.hpp>
""")
# Load the library
gSystem.Load("/usr/local/lib/libEtaler.so");

# make a handy namespace alias
et = ROOT.et
```

Then you can call Etaler functions from Python.

```Python
s = et.Shape();
[s.push_back(v) for v in [2, 2]] #HACK: ROOT doesn't support initalizer lists.
t = et.ones(s)
print(t)
"""
{{ 1, 1}, 
 { 1, 1}}

"""
```

## PyEtaler
The offical Python binding - [PyEtaler](https://guthub.com/etaler/pyetaler) in currently work in progress. But we recomment using ROOT to bind from Python before PyEtaler leaves WIP.
