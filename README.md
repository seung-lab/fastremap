[![Build Status](https://travis-ci.org/seung-lab/fastremap.svg?branch=master)](https://travis-ci.org/seung-lab/fastremap) [![PyPI version](https://badge.fury.io/py/fastremap.svg)](https://badge.fury.io/py/fastremap)  

# fastremap
Relabel integers in a numpy array based on dicts, arrays, and sequential renumbering from one. This module seems trivial, but it's necessary. Python loops are slow, so often numpy is used to perform remapping on large arrays (hundreds of megabytes or gigabytes). However, in order to efficiently remap an array in numpy you need a key-value array where the index is the key and the value is the contents of that index. If there are 32 or 64 bit labels in the array, this becomes impractical despite the triviality of the operation. It's conceivable to use numba for this, but having a cython library as an option can be convenient.

Available functions:  
- renumber: Relabel array from 1 to N which can often use smaller datatypes.
- remap: Custom relabeling of values in an array from a dictionary.
- remap_from_array: Same as remap, but the map is an array where the key is the array index and the value is the value.
- remap_from_array_kv: Same as remap, but the map consists of two equal sized arrays, the first containing keys, the second containing values.

```python
import fastremap
import numpy as np

arr = np.array(..., dtype=np.int64) # array contains 500 unique labels

# Renumber labels from 1 to 501 and preserve 0
arr = fastremap.renumber(arr, preserve_zero=True) # Returns uint16 array (smallest possible)
arr = fastremap.renumber(arr, preserve_zero=False) # Returns uint16 array, contains [1,502]

mappings = {
  1: 100,
  2: 200,
  -3: 7,
}

arr = np.array([5, 1, 2, -5, -3, 10, 6])
# Custom remapping of -3, 5, and 6 leaving the rest alone
arr = fastremap.remap(arr, mappings, preserve_missing_labels=True) 
# result: [ 5, 100, 200, -5, 7, 10, 6 ]

try:
  arr = fastremap.remap(arr, mappings, preserve_missing_labels=False) 
except KeyError:
  # When preserve_missing_labels is False, a KeyError is thrown if a
  # value is encountered that isn't in mappings.
  pass 

arr = np.array([1,2,3,3,2])
kvs = np.array([0,10,20,30]) # key is array index, value is value
result = fastremap.remap_from_array(arr, kvs)
# result: [10, 20, 30, 30, 20]

arr = np.array([1,2,3,3,2])
keys = np.array([3,4,2,1]) # key is array index, value is value
vals = np.array([10,20,30,40]) # key is array index, value is value
result = fastremap.remap_from_array_kv(arr, keys, vals)
# result: [40, 30, 10, 10, 30]
```

## `pip` Installation

*If binaries are available for your system.*

```bash
pip install fastremap
```

*If not, a C++ compiler is required.*

```bash
pip install numpy
pip install fastremap
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

--  
Made with <3



