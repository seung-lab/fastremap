[![Build Status](https://travis-ci.org/seung-lab/fastremap.svg?branch=master)](https://travis-ci.org/seung-lab/fastremap) [![PyPI version](https://badge.fury.io/py/fastremap.svg)](https://badge.fury.io/py/fastremap)  

# fastremap

Renumber and relabel Numpy arrays at C++ speed and physically convert rectangular Numpy arrays between C and Fortran order using an in-place transposition.   

## `pip` Installation

```bash
pip install fastremap
```

*If not, a C++ compiler is required.*

```bash
pip install numpy
pip install fastremap --no-binary :all:
```

## Manual Installation

*A C++ compiler is required.*

```bash
sudo apt-get install g++ python3-dev 
mkvirtualenv -p python3 fastremap
pip install numpy

# Choose one:
python setup.py develop  
python setup.py install 
```

## The Problem of Remapping

Python loops are slow, so Numpy is often used to perform remapping on large arrays (hundreds of megabytes or gigabytes). In order to efficiently remap an array in Numpy you need a key-value array where the index is the key and the value is the contents of that index. 

```python 
import numpy as np 

original = np.array([ 1, 3, 5, 5, 10 ])
remap = np.array([ 0, -5, 0, 6, 0, 0, 2, 0, 0, 0, -100 ])
# Keys:            0   1  2  3  4  5  6  7  8  9    10

remapped = remap[ original ]
>>> [ -5, 6, 2, 2, -100 ]
```

If there are 32 or 64 bit labels in the array, this becomes impractical as the size of the array can grow larger than RAM. Therefore, it would be helpful to be able to perform this mapping using a C speed loop. Numba can be used for this in some circumstances. However, this library provides an alternative.

```python
import numpy as np
import fastremap 

mappings = {
  1: 100,
  2: 200,
  -3: 7,
}

arr = np.array([5, 1, 2, -5, -3, 10, 6])
# Custom remapping of -3, 5, and 6 leaving the rest alone
arr = fastremap.remap(arr, mappings, preserve_missing_labels=True) 
# result: [ 5, 100, 200, -5, 7, 10, 6 ]
```

## The Problem of Renumbering 

Sometimes a 64-bit array contains values that could be represented by an 8-bit array. However, similarly to the remapping problem, Python loops can be too slow to do this. Numpy doesn't provide a convenient way to do it either. Therefore this library provides an alternative solution.

```python
import fastremap
import numpy as np

arr = np.array([ 283732875, 439238823, 283732875, 182812404, 0 ], dtype=np.int64) 

arr, remapping = fastremap.renumber(arr, preserve_zero=True) # Returns uint8 array
>>> arr = [ 1, 2, 1, 3, 0 ]
>>> remapping = { 0: 0, 283732875: 1, 439238823: 2, 182812404: 3 }

arr, remapping = fastremap.renumber(arr, preserve_zero=False) # Returns uint8 array
>>> arr = [ 1, 2, 1, 3, 4 ]
>>> remapping = { 0: 4, 283732875: 1, 439238823: 2, 182812404: 3 }

arr, remapping = fastremap.renumber(arr, preserve_zero=False, in_place=True) # Mutate arr to use less memory
>>> arr = [ 1, 2, 1, 3, 4 ]
>>> remapping = { 0: 4, 283732875: 1, 439238823: 2, 182812404: 3 }
```

## The Problem of In-Place Transposition 

When transitioning between different media, e.g. CPU to GPU, CPU to Network, CPU to disk, it's often necessary to physically transpose multi-dimensional arrays to reformat as C or Fortran order. Tranposing matrices is also a common action in linear algebra, but often you can get away with just changing the strides.

An out-of-place transposition is easy to write, and often faster, but it will spike peak memory consumption. This library grants the user the option of performing an in-place transposition which trades CPU time for peak memory usage. In the special case of square or cubic arrays, the in-place transpisition is both lower memory and faster.

- **fastremap.asfortranarray:** Same as np.asfortranarray but will perform the transposition in-place for 1, 2, 3, and 4D arrays. 2D and 3D square matrices are faster to process than with Numpy.
- **fastremap.ascontiguousarray:** Same as np.ascontiguousarray but will perform the transposition in-place for 1, 2, 3, and 4D arrays. 2D and 3D square matrices are faster to process than with Numpy.

```python
import fastremap
import numpy as np 

arr = np.ones((512,512,512), dtype=np.float32)
arr = fastremap.asfortranarray(x)

arr = np.ones((512,512,512), dtype=np.float32, order='F')
arr = fastremap.ascontiguousarray(x)
```

## All Available Functions:  
- **renumber:** Relabel array from 1 to N which can often use smaller datatypes.
- **remap:** Custom relabeling of values in an array from a dictionary.
- **mask:** Zero out labels in an array specified by a given list.
- **mask_except**: Zero out all labels except those specified in a given list.
- **remap_from_array:** Same as remap, but the map is an array where the key is the array index and the value is the value.
- **remap_from_array_kv:** Same as remap, but the map consists of two equal sized arrays, the first containing keys, the second containing values.
- **asfortranarray:** Perform an in-place matrix transposition for rectangular arrays if memory is contiguous, standard numpy otherwise.
- **ascontiguousarray:** Perform an in-place matrix transposition for rectangular arrays if memory is contiguous, standard numpy algorithm otherwise.

## C++ Usage

The in-place matrix transposition is implemented in ipt.hpp. If you're working in C++, you can also use it directly like so:

```cpp
#include "ipt.hpp"

int main() {

  int sx = 128;
  int sy = 124;
  int sz = 103;
  int sw = 3;

  auto* arr = ....;

  // All primitive number types supported
  // The array will be modified in place, 
  // so these functions are void type.
  ipt::ipt<int>(arr, sx, sy);            // 2D
  ipt::ipt<float>(arr, sx, sy, sz);      // 3D
  ipt::ipt<double>(arr, sx, sy, sz, sw); // 4D

  return 0;
}
```

--  
Made with <3



