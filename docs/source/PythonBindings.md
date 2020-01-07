# Python bindings

## PyEtaler
[PyEtaler](https://github.com/etaler/pyetaler) is the offical binding for Etaler. We try to keep the Python API as close to the C++ one as possible. So you can use the C++ document as the Python document. With that said, some functions are changed in the binding to make it more Pythonic.

```python
>>> from etaler import et
>>> et.ones([2, 2])
{{ 1, 1}, 
 { 1, 1}}
```

## ROOT

If cppyy is not avaliable to you for any reason. You can use Etaler in Python via [ROOT](https://root.cern.ch) and it's automatic binding generation feature.

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
s = et.Shape()
[s.push_back(v) for v in [2, 2]] #HACK: ROOT doesn't support initalizer lists.
t = et.ones(s)
print(t)
"""
{{ 1, 1}, 
 { 1, 1}}
"""
```