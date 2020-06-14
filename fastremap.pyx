"""
Functions related to remapping image volumes.

Renumber volumes into smaller data types, mask out labels
or their complement, and remap the values of image volumes.

This module also constains the facilities for performing
and in-place matrix transposition for up to 4D arrays. This is 
helpful for converting between C and Fortran order in memory
constrained environments when format shifting.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018 - January 2020
"""

cimport cython
from libc.stdint cimport (  
  uint8_t, uint16_t, uint32_t, uint64_t,
  int8_t, int16_t, int32_t, int64_t
)
from libcpp.unordered_map cimport unordered_map

from collections import defaultdict
from functools import reduce
import operator

import numpy as np
cimport numpy as cnp

__version__ = '1.10.1'
__VERSION__ = __version__

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

ctypedef fused ALLINT:
  UINT
  int8_t
  int16_t
  int32_t
  int64_t

ctypedef fused ALLINT_2:
  ALLINT

ctypedef fused NUMBER:
  ALLINT
  float
  double

ctypedef fused COMPLEX_NUMBER:
  NUMBER
  float complex 

cdef extern from "ipt.hpp" namespace "pyipt":
  cdef void _ipt2d[T](T* arr, size_t sx, size_t sy)
  cdef void _ipt3d[T](
    T* arr, size_t sx, size_t sy, size_t sz
  )
  cdef void _ipt4d[T](
    T* arr, size_t sx, size_t sy, size_t sz, size_t sw
  )

def argmin(arr):
  """
  Returns an index into a flattened array containing 
  a minimum value of the array. 
  """
  if arr.flags['C_CONTIGUOUS']:
    return np.argmin(arr)
  return argminmax(arr)[0]

def argmax(arr):
  """
  Returns an index into a flattened array containing 
  a maximum value of the array. 
  """
  if arr.flags['C_CONTIGUOUS']:
    return np.argmax(arr)
  return argminmax(arr)[1]

def argminmax(arr):
  """
  Returns (argmin(arr), argmax(arr)) computed in a single pass.
  Returns (None, None) if array is size zero.
  """
  res = _argminmax(reshape(arr, (arr.size,)))
  if res == (None, None):
    return res

  if arr.flags['C_CONTIGUOUS']:
    return res

  lt = np.unravel_index(res[0], arr.shape, order='F')
  rt = np.unravel_index(res[1], arr.shape, order='F')

  return ravel_index(lt, arr.shape), ravel_index(rt, arr.shape)

def _argminmax(cnp.ndarray[NUMBER, ndim=1] arr):
  cdef size_t i = 0
  cdef size_t size = arr.size

  if size == 0:
    return (None, None)

  cdef NUMBER minval = arr[0]
  cdef size_t minidx = 0
  cdef NUMBER maxval = arr[0]
  cdef size_t maxidx = 0

  for i in range(1, size):
    if minval > arr[i]:
      minval = arr[i]
      minidx = i
    if maxval < arr[i]:
      maxval = arr[i]
      maxidx = i

  return (minidx, maxidx)

def ravel_index(coord, shape, order='C'):
  """
  Counterpart to np.unravel_index. 

  Takes a coordinate and converts it into the
  linear array index with the corresponding shape
  in either C or Fortran order.

  Returns: int
  """
  cdef size_t idx = 0
  cdef size_t mult = 1
  
  cdef size_t i = 0
  cdef size_t size = 1

  if order == 'C':
    shape = shape[::-1]
    coord = coord[::-1]

  for i, size in enumerate(shape):
    idx += mult * coord[i]
    mult *= size

  return idx

def minmax(arr):
  """
  Returns (min(arr), max(arr)) computed in a single pass.
  Returns (None, None) if array is size zero.
  """
  return _minmax(reshape(arr, (arr.size,)))

def _minmax(cnp.ndarray[NUMBER, ndim=1] arr):
  cdef size_t i = 0
  cdef size_t size = arr.size

  if size == 0:
    return None, None

  cdef NUMBER minval = arr[0]
  cdef NUMBER maxval = arr[0]

  for i in range(1, size):
    if minval > arr[i]:
      minval = arr[i]
    if maxval < arr[i]:
      maxval = arr[i]

  return minval, maxval

def renumber(arr, start=1, preserve_zero=True, in_place=False):
  """
  renumber(arr, start=1, preserve_zero=True, in_place=False)

  Given an array of integers, renumber all the unique values starting
  from 1. This can allow us to reduce the size of the data width required
  to represent it.

  arr: A numpy array
  start (default: 1): Start renumbering from this value
  preserve_zero (default: True): Don't renumber zero.
  in_place (default: False): Perform the renumbering in-place to avoid
    an extra copy. This option depends on a fortran or C contiguous
    array. A copy will be made if the array is not contiguous.

  Return: a renumbered array, dict with remapping of oldval => newval
  """
  if arr.size == 0:
    return arr, {}

  if arr.dtype == np.bool and preserve_zero:
    return arr, { 0: 0, 1: start }
  elif arr.dtype == np.bool:
    arr = arr.view(np.uint8)

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  shape = arr.shape
  order = 'F' if arr.flags['F_CONTIGUOUS'] else 'C'
  in_place = in_place and (arr.flags['F_CONTIGUOUS'] or arr.flags['C_CONTIGUOUS'])

  if not in_place:
    arr = np.copy(arr, order=order)

  arr = np.lib.stride_tricks.as_strided(arr, shape=(arr.size,), strides=(nbytes,))
  arr, remap_dict = _renumber(arr, <int64_t>start, preserve_zero)
  arr = reshape(arr, shape, order)

  return arr, remap_dict

def reshape(arr, shape, order=None):
  """
  If the array is contiguous, attempt an in place reshape
  rather than potentially making a copy.

  Required:
    arr: The input numpy array.
    shape: The desired shape (must be the same size as arr)

  Optional: 
    order: 'C', 'F', or None (determine automatically)

  Returns: reshaped array
  """
  if order is None:
    if arr.flags['F_CONTIGUOUS']:
      order = 'F'
    elif arr.flags['C_CONTIGUOUS']:
      order = 'C'
    else:
      return arr.reshape(shape)

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  if order == 'C':
    strides = [ reduce(operator.mul, shape[i:]) * nbytes for i in range(1, len(shape)) ]
    strides += [ nbytes ]
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
  else:
    strides = [ reduce(operator.mul, shape[:i]) * nbytes for i in range(1, len(shape)) ]
    strides = [ nbytes ] + strides
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _renumber(cnp.ndarray[NUMBER, cast=True, ndim=1] arr, int64_t start=1, preserve_zero=True):
  """
  renumber(arr, int64_t start=1, preserve_zero=True)

  Given an array of integers, renumber all the unique values starting
  from 1. This can allow us to reduce the size of the data width required
  to represent it.

  arr: A numpy array
  start (default: 1): Start renumbering from this value
  preserve_zero (default ): 

  Return: a renumbered array, dict with remapping of oldval => newval
  """
  cdef unordered_map[NUMBER, NUMBER] remap_dict
  if preserve_zero:
    remap_dict[0] = 0

  cdef NUMBER[:] arrview = arr

  cdef NUMBER remap_id = start
  cdef NUMBER elem

  cdef NUMBER last_elem = 0
  cdef NUMBER last_remap_id = 0
  if not preserve_zero:
    last_remap_id = start

  cdef size_t size = arr.size
  cdef size_t i = 0

  for i in range(size):
    elem = arrview[i]

    if elem == last_elem:
      arrview[i] = last_remap_id
      continue

    if remap_dict.find(elem) == remap_dict.end():
      arrview[i] = remap_id
      remap_dict[elem] = remap_id
      remap_id += 1      
    else:
      arrview[i] = remap_dict[elem]

    last_elem = elem 
    last_remap_id = arrview[i]

  factor = remap_id 
  if abs(start) > abs(factor):
    factor = start

  return refit(arr, factor), remap_dict

def refit(arr, value=None, increase_only=False, exotics=False):
  """
  Resize the array to the smallest dtype of the 
  same kind that will fit a given value.

  For example, if the input array is uint8 and 
  the value is 2^20 return the array as a 
  uint32.

  Works for standard floating, integer, 
  unsigned integer, and complex types.

  arr: numpy array
  value: value to fit array to. if None,
    it is set to the value of the absolutely
    larger of the min and max value in the array.
  increase_only: if true, only resize the array if it can't
    contain value. if false, always resize to the 
    smallest size that fits.
  exotics: if true, allow e.g. half precision floats (16-bit) 
    or double complex (128-bit)

  Return: refitted array
  """

  if value is None:
    min_value, max_value = minmax(arr)
    if min_value is None or max_value is None:
      min_value = 0 
      max_value = 0

    if abs(max_value) > abs(min_value):
      value = max_value
    else:
      value = min_value

  dtype = fit_dtype(arr.dtype, value, exotics=exotics)

  if increase_only and np.dtype(dtype).itemsize <= np.dtype(arr.dtype).itemsize:
    return arr
  elif dtype == arr.dtype:
    return arr
  return arr.astype(dtype)

def fit_dtype(dtype, value, exotics=False):
  """
  Find the smallest dtype of the 
  same kind that will fit a given value.

  For example, if the input array is uint8 and 
  the value is 2^20 return the array as a 
  uint32.

  Works for standard floating, integer, 
  unsigned integer, and complex types.

  exotics: if True, allow fitting to
    e.g. float16 (half-precision, 16-bits) 
      or double complex (which takes 128-bits).

  Return: refitted array
  """
  dtype = np.dtype(dtype)
  if np.issubdtype(dtype, np.floating):
    if exotics:
      sequence = [ np.float16, np.float32, np.float64 ] 
    else:
      sequence = [ np.float32, np.float64 ] 
    infofn = np.finfo
  elif np.issubdtype(dtype, np.unsignedinteger):
    sequence = [ np.uint8, np.uint16, np.uint32, np.uint64 ]
    infofn = np.iinfo
    if value < 0:
      raise ValueError(str(value) + " is negative but unsigned data type {} is selected.".format(dtype))
  elif np.issubdtype(dtype, np.complexfloating):
    if exotics:
      sequence = [ np.csingle, np.cdouble ]
    else:
      sequence = [ np.csingle ]
    infofn = np.finfo
  elif np.issubdtype(dtype, np.integer):
    sequence = [ np.int8, np.int16, np.int32, np.int64 ]
    infofn = np.iinfo
  else:
    raise ValueError(
      "Unsupported dtype: {} Only standard floats, integers, and complex types are supported.".format(dtype)
    )

  test_value = np.real(value) 
  if abs(np.real(value)) < abs(np.imag(value)):
    test_value = np.imag(value)

  for seq_dtype in sequence:
    if test_value >= 0 and infofn(seq_dtype).max >= test_value:
      return seq_dtype
    elif test_value < 0 and infofn(seq_dtype).min <= test_value:
      return seq_dtype

  raise ValueError("Unable to find a compatible dtype for {} that can fit {}".format(
    dtype, value
  ))

def mask(arr, labels, in_place=False, value=0):
  """
  mask(arr, labels, in_place=False, value=0)

  Mask out designated labels in an array with the
  given value. 

  Alternative implementation of:

  arr[np.isin(labels)] = value

  arr: an N-dimensional numpy array
  labels: an iterable list of integers
  in_place: if True, modify the input array to reduce
    memory consumption.
  value: mask value

  Returns: arr with `labels` masked out
  """
  labels = { lbl: value for lbl in labels }
  return remap(arr, labels, preserve_missing_labels=True, in_place=in_place)

def mask_except(arr, labels, in_place=False, value=0):
  """
  mask_except(arr, labels, in_place=False, value=0)

  Mask out all labels except the provided list.

  Alternative implementation of:

  arr[~np.isin(labels)] = value

  arr: an N-dimensional numpy array
  labels: an iterable list of integers
  in_place: if True, modify the input array to reduce
    memory consumption.
  value: mask value

  Returns: arr with all labels except `labels` masked out
  """
  shape = arr.shape 

  if arr.flags['F_CONTIGUOUS']:
    order = 'F'
  else:
    order = 'C'

  if not in_place:
    arr = np.copy(arr, order=order)

  arr = reshape(arr, (arr.size,))
  arr = _mask_except(arr, labels, value)
  return reshape(arr, shape, order=order)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)          
def _mask_except(cnp.ndarray[ALLINT] arr, list labels, ALLINT value):
  cdef ALLINT[:] arrview = arr
  cdef size_t i = 0
  cdef size_t size = arr.size

  if size == 0:
    return arr

  cdef unordered_map[ALLINT, ALLINT] tbl 

  for label in labels:
    tbl[label] = label 

  cdef ALLINT last_elem = arrview[0]
  cdef ALLINT last_elem_value = 0

  if tbl.find(last_elem) == tbl.end():
    last_elem_value = value
  else:
    last_elem_value = last_elem

  for i in range(size):
    if arrview[i] == last_elem:
      arrview[i] = last_elem_value
    elif tbl.find(arrview[i]) == tbl.end():
      last_elem = arrview[i]
      last_elem_value = value
      arrview[i] = value
    else:
      last_elem = arrview[i]
      last_elem_value = arrview[i]

  return arr

def component_map(component_labels, parent_labels):
  """
  Given two sets of images that have a surjective mapping between their labels,
  generate a dictionary for that mapping.

  For example, generate a mapping from connected components of labels to their
  parent labels.

  e.g. component_map([ 1, 2, 3, 4 ], [ 5, 5, 6, 7 ])
    returns { 1: 5, 2: 5, 3: 6, 4: 7 }

  Returns: { $COMPONENT_LABEL: $PARENT_LABEL }
  """
  if not isinstance(component_labels, np.ndarray):
    component_labels = np.array(component_labels)
  if not isinstance(parent_labels, np.ndarray):
    parent_labels = np.array(parent_labels)

  if component_labels.size == 0:
    return {}

  if component_labels.shape != parent_labels.shape:
    raise ValueError("The size of the inputs must match: {} vs {}".format(
      component_labels.shape, parent_labels.shape
    ))

  shape = component_labels.shape 

  component_labels = reshape(component_labels, (component_labels.size,))
  parent_labels = reshape(parent_labels, (parent_labels.size,))
  return _component_map(component_labels, parent_labels)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _component_map(
  cnp.ndarray[ALLINT, ndim=1, cast=True] component_labels, 
  cnp.ndarray[ALLINT_2, ndim=1, cast=True] parent_labels
):
  cdef size_t size = component_labels.size
  if size == 0:
    return {}

  cdef dict remap = {}
  cdef size_t i = 0

  cdef ALLINT last_label = component_labels[0]
  remap[component_labels[0]] = parent_labels[0]
  for i in range(size):
    if last_label == component_labels[i]:
      continue
    remap[component_labels[i]] = parent_labels[i]
    last_label = component_labels[i]

  return remap

def inverse_component_map(parent_labels, component_labels):
  """
  Given two sets of images that have a mapping between their labels,
  generate a dictionary for that mapping.

  For example, generate a mapping from connected components of labels to their
  parent labels.

  e.g. inverse_component_map([ 1, 2, 1, 3 ], [ 4, 4, 5, 6 ])
    returns { 1: [ 4, 5 ], 2: [ 4 ], 3: [ 6 ] }

  Returns: { $PARENT_LABEL: [ $COMPONENT_LABELS, ... ] }
  """
  if not isinstance(component_labels, np.ndarray):
    component_labels = np.array(component_labels)
  if not isinstance(parent_labels, np.ndarray):
    parent_labels = np.array(parent_labels)

  if component_labels.size == 0:
    return {}

  if component_labels.shape != parent_labels.shape:
    raise ValueError("The size of the inputs must match: {} vs {}".format(
      component_labels.shape, parent_labels.shape
    ))

  shape = component_labels.shape 

  component_labels = reshape(component_labels, (component_labels.size,))
  parent_labels = reshape(parent_labels, (parent_labels.size,))
  return _inverse_component_map(parent_labels, component_labels)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _inverse_component_map(
  cnp.ndarray[ALLINT, ndim=1, cast=True] parent_labels, 
  cnp.ndarray[ALLINT_2, ndim=1, cast=True] component_labels
):
  cdef size_t size = parent_labels.size
  if size == 0:
    return {}

  remap = defaultdict(list)
  cdef size_t i = 0

  cdef ALLINT last_label = parent_labels[0]
  remap[parent_labels[0]].append(component_labels[0])
  for i in range(size):
    if last_label == parent_labels[i]:
      continue
    remap[parent_labels[i]].append(component_labels[i])
    last_label = parent_labels[i]

  return remap

def remap(arr, table, preserve_missing_labels=False, in_place=False):
  """
  remap(cnp.ndarray[COMPLEX_NUMBER] arr, dict table, 
    preserve_missing_labels=False, in_place=False)

  Remap an input numpy array in-place according to the values in the given 
  dictionary "table".   

  arr: an N-dimensional numpy array
  table: { label: new_label_value, ... }
  preserve_missing_labels: If an array value is not present in "table"...
    True: Leave it alone.
    False: Throw a KeyError.
  in_place: if True, modify the input array to reduce
    memory consumption.

  Returns: remapped array
  """
  if type(arr) == list:
    arr = np.array(arr)

  shape = arr.shape 

  if arr.flags['F_CONTIGUOUS']:
    order = 'F'
  else:
    order = 'C'

  original_dtype = arr.dtype
  if len(table):
    min_label, max_label = min(table.values()), max(table.values())
    fit_value = min_label if abs(min_label) > abs(max_label) else max_label
    arr = refit(arr, fit_value, increase_only=True)

  if not in_place and original_dtype == arr.dtype:
    arr = np.copy(arr, order=order)

  arr = reshape(arr, (arr.size,))
  arr = _remap(arr, table, preserve_missing_labels)
  return reshape(arr, shape, order=order)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)          
def _remap(cnp.ndarray[NUMBER] arr, dict table, uint8_t preserve_missing_labels):
  cdef NUMBER[:] arrview = arr
  cdef size_t i = 0
  cdef size_t size = arr.size
  cdef NUMBER elem = 0

  if size == 0:
    return arr

  cdef unordered_map[NUMBER, NUMBER] tbl 

  for k, v in table.items():
    tbl[k] = v 

  cdef NUMBER last_elem = arrview[0]
  cdef NUMBER last_remap_id = 0
  
  with nogil:
    if tbl.find(last_elem) == tbl.end():
      if not preserve_missing_labels:
        raise KeyError("{} was not in the remap table.".format(last_elem))  
    else:
      arrview[0] = tbl[last_elem]
      last_remap_id = arrview[0]

    for i in range(1, size):
      elem = arrview[i]

      if elem == last_elem:
        arrview[i] = last_remap_id
        continue

      if tbl.find(elem) == tbl.end():
        if preserve_missing_labels:
          last_elem = elem
          last_remap_id = elem
          continue
        else:
          raise KeyError("{} was not in the remap table.".format(elem))  
      else:
        arrview[i] = tbl[elem]
      
      last_elem = elem
      last_remap_id = arrview[i]

  return arr

@cython.boundscheck(False)
def remap_from_array(cnp.ndarray[UINT] arr, cnp.ndarray[UINT] vals):
  """
  remap_from_array(cnp.ndarray[UINT] arr, cnp.ndarray[UINT] vals)
  """
  cdef UINT[:] valview = vals
  cdef UINT[:] arrview = arr

  cdef size_t i = 0
  cdef size_t size = arr.size 
  cdef size_t maxkey = vals.size - 1
  cdef UINT elem

  with nogil:
    for i in range(size):
      elem = arr[i]
      if elem < 0 or elem > maxkey:
        continue
      arrview[i] = vals[elem]

  return arr

@cython.boundscheck(False)
def remap_from_array_kv(cnp.ndarray[ALLINT] arr, cnp.ndarray[ALLINT] keys, cnp.ndarray[ALLINT] vals):
  """
  remap_from_array_kv(cnp.ndarray[ALLINT] arr, cnp.ndarray[ALLINT] keys, cnp.ndarray[ALLINT] vals)
  """
  cdef ALLINT[:] keyview = keys
  cdef ALLINT[:] valview = vals
  cdef ALLINT[:] arrview = arr
  cdef unordered_map[ALLINT, ALLINT] remap_dict

  assert keys.size == vals.size

  cdef size_t i = 0
  cdef size_t size = keys.size 
  cdef ALLINT elem

  with nogil:
    for i in range(size):
      remap_dict[keys[i]] = vals[i]

  i = 0
  size = arr.size 

  with nogil:
    for i in range(size):
      elem = arr[i]
      if remap_dict.find(elem) == remap_dict.end():
        continue
      else:
          arrview[i] = remap_dict[elem]

  return arr

def pixel_pairs(labels):
  """
  Computes the number of matching adjacent memory locations.

  This is useful for rapidly evaluating whether an image is
  more binary or more connectomics like.
  """
  if labels.size == 0:
    return 0
  return _pixel_pairs(reshape(labels, (labels.size,)))

def _pixel_pairs(cnp.ndarray[ALLINT, ndim=1] labels):
  cdef size_t voxels = labels.size

  cdef size_t pairs = 0
  cdef ALLINT label = labels[0]

  cdef size_t i = 0
  for i in range(1, voxels):
    if label == labels[i]:
      pairs += 1
    else:
      label = labels[i]

  return pairs

def unique(labels, return_index=False, return_inverse=False, return_counts=False, axis=None):
  """
  Compute the sorted set of unique labels in the input array.

  return_counts: also return the unique label frequency as an array.

  Returns: 
    if return_counts:
      return (unique_labels, unique_counts)
    else:
      return unique_labels
  """
  if not np.issubdtype(labels.dtype, np.integer):
    raise TypeError("fastremap.unique only supports integer types.")

  # These flags are currently unsupported so call uncle and
  # use the standard implementation instead.
  if return_index or return_inverse or (axis is not None):
    return np.unique(
      labels, 
      return_index=return_index, 
      return_inverse=return_inverse, 
      return_counts=return_counts, 
      axis=axis
    )

  cdef size_t voxels = labels.size

  shape = labels.shape
  labels = reshape(labels, (voxels,))

  cdef int64_t max_label
  cdef int64_t min_label
  min_label, max_label = minmax(labels)

  if voxels == 0:
    uniq = np.array([], dtype=labels.dtype)
    counts = np.array([], dtype=np.uint32)
  elif min_label >= 0 and max_label < <int64_t>voxels:
    uniq, counts = unique_via_array(labels, max_label)
  elif (max_label - min_label) <= <int64_t>voxels:
    uniq, counts = unique_via_shifted_array(labels, min_label, max_label)
  elif float(pixel_pairs(labels)) / float(voxels) > 0.66:
    uniq, counts = unique_via_renumber(labels)
  else:
    uniq, counts = unique_via_sort(labels)

  if return_counts:
    return uniq, counts
  return uniq

def unique_via_shifted_array(labels, min_label=None, max_label=None):
  if min_label is None or max_label is None:
    min_label, max_label = minmax(labels)

  labels -= min_label
  uniq, counts = unique_via_array(labels, max_label - min_label + 1)
  labels += min_label
  uniq += min_label
  return uniq, counts

def unique_via_renumber(labels):
  dtype = labels.dtype
  labels, remap = renumber(labels)
  remap = { v:k for k,v in remap.items() }
  uniq, counts = unique_via_array(labels, max(remap.keys()))
  uniq = np.array([ remap[segid] for segid in uniq ], dtype=dtype)
  return uniq, counts

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def unique_via_sort(cnp.ndarray[ALLINT, ndim=1] labels):
  """Slower than unique_via_array but can handle any label."""
  labels = np.copy(labels)
  labels.sort()

  cdef size_t voxels = labels.size  

  cdef cnp.ndarray[ALLINT, ndim=1] uniq = np.zeros((voxels,), dtype=labels.dtype)
  cdef cnp.ndarray[uint32_t, ndim=1] counts = np.zeros((voxels,), dtype=np.uint32)

  cdef size_t i = 0
  cdef size_t j = 0

  cdef ALLINT cur = labels[0]
  cdef size_t accum = 1
  for i in range(1, voxels):
    if cur == labels[i]:
      accum += 1
    else:
      uniq[j] = cur
      counts[j] = accum
      accum = 1
      cur = labels[i]
      j += 1

  uniq[j] = cur
  counts[j] = accum

  return uniq[:j+1], counts[:j+1]

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def unique_via_array(cnp.ndarray[ALLINT, ndim=1] labels, size_t max_label):
  """
  unique(cnp.ndarray[ALLINT, ndim=1] labels, return_counts=False)

  Faster implementation of np.unique that depends
  on the maximum label in the array being less than
  the size of the array.
  """
  cdef cnp.ndarray[uint32_t, ndim=1] counts = np.zeros( 
    (max_label+1,), dtype=np.uint32
  )

  cdef size_t voxels = labels.shape[0]
  cdef size_t i = 0
  for i in range(voxels):
    counts[labels[i]] += 1

  cdef size_t real_size = 0
  for i in range(max_label + 1):
    if counts[i] > 0:
      real_size += 1

  cdef cnp.ndarray[ALLINT, ndim=1] segids = np.zeros( 
    (real_size,), dtype=labels.dtype
  )
  cdef cnp.ndarray[uint32_t, ndim=1] cts = np.zeros( 
    (real_size,), dtype=np.uint32
  )

  cdef size_t j = 0
  for i in range(max_label + 1):
    if counts[i] > 0:
      segids[j] = i
      cts[j] = counts[i]
      j += 1

  return segids, cts
  
def transpose(arr):
  """
  transpose(arr)

  For up to four dimensional matrices, perform in-place transposition. 
  Square matrices up to three dimensions are faster than numpy's out-of-place
  algorithm. Default to the out-of-place implementation numpy uses for cases
  that aren't specially handled.

  Returns: transposed numpy array
  """
  if not arr.flags['F_CONTIGUOUS'] and not arr.flags['C_CONTIGUOUS']:
    arr = np.copy(arr, order='C')

  shape = arr.shape
  strides = arr.strides

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  dtype = arr.dtype
  if arr.dtype == np.bool:
    arr = arr.view(np.uint8)

  if arr.ndim == 2:
    arr = ipt2d(arr)
    return arr.view(dtype)
  elif arr.ndim == 3:
    arr = ipt3d(arr)
    return arr.view(dtype)
  elif arr.ndim == 4:
    arr = ipt4d(arr)
    return arr.view(dtype)
  else:
    return arr.T

def asfortranarray(arr):
  """
  asfortranarray(arr)

  For up to four dimensional matrices, perform in-place transposition. 
  Square matrices up to three dimensions are faster than numpy's out-of-place
  algorithm. Default to the out-of-place implementation numpy uses for cases
  that aren't specially handled.

  Returns: transposed numpy array
  """
  if arr.flags['F_CONTIGUOUS']:
    return arr
  elif not arr.flags['C_CONTIGUOUS']:
    return np.asfortranarray(arr)
  elif arr.ndim == 1:
    return arr 

  shape = arr.shape
  strides = arr.strides

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  dtype = arr.dtype
  if arr.dtype == np.bool:
    arr = arr.view(np.uint8)

  if arr.ndim == 2:
    arr = ipt2d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(nbytes, shape[0] * nbytes))
    return arr.view(dtype)
  elif arr.ndim == 3:
    arr = ipt3d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(nbytes, shape[0] * nbytes, shape[0] * shape[1] * nbytes))
    return arr.view(dtype)
  elif arr.ndim == 4:
    arr = ipt4d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, 
      strides=(
        nbytes, 
        shape[0] * nbytes, 
        shape[0] * shape[1] * nbytes, 
        shape[0] * shape[1] * shape[2] * nbytes
      ))
    return arr.view(dtype)
  else:
    return np.asfortranarray(arr)

def ascontiguousarray(arr):
  """
  ascontiguousarray(arr)

  For up to four dimensional matrices, perform in-place transposition. 
  Square matrices up to three dimensions are faster than numpy's out-of-place
  algorithm. Default to the out-of-place implementation numpy uses for cases
  that aren't specially handled.

  Returns: transposed numpy array
  """
  if arr.flags['C_CONTIGUOUS']:
    return arr
  elif not arr.flags['F_CONTIGUOUS']:
    return np.ascontiguousarray(arr)
  elif arr.ndim == 1:
    return arr 

  shape = arr.shape
  strides = arr.strides

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  dtype = arr.dtype
  if arr.dtype == np.bool:
    arr = arr.view(np.uint8)

  if arr.ndim == 2:
    arr = ipt2d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(shape[1] * nbytes, nbytes))
    return arr.view(dtype)
  elif arr.ndim == 3:
    arr = ipt3d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(
        shape[2] * shape[1] * nbytes, 
        shape[2] * nbytes, 
        nbytes,
      ))
    return arr.view(dtype)
  elif arr.ndim == 4:
    arr = ipt4d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, 
      strides=(
        shape[3] * shape[2] * shape[1] * nbytes,
        shape[3] * shape[2] * nbytes, 
        shape[3] * nbytes, 
        nbytes, 
      ))
    return arr.view(dtype)
  else:
    return np.ascontiguousarray(arr)

def ipt2d(cnp.ndarray[COMPLEX_NUMBER, cast=True, ndim=2] arr):
  cdef COMPLEX_NUMBER[:,:] arrview = arr

  cdef size_t sx
  cdef size_t sy

  if arr.flags['F_CONTIGUOUS']:
    sx = arr.shape[0]
    sy = arr.shape[1]
  else:
    sx = arr.shape[1]
    sy = arr.shape[0]

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  # ipt doesn't do anything with values, 
  # just moves them around, so only bit width matters
  # int, uint, float, bool who cares
  if nbytes == 1:
    _ipt2d[uint8_t](
      <uint8_t*>&arrview[0,0],
      sx, sy
    )
  elif nbytes == 2:
    _ipt2d[uint16_t](
      <uint16_t*>&arrview[0,0],
      sx, sy
    )
  elif nbytes == 4:
    _ipt2d[uint32_t](
      <uint32_t*>&arrview[0,0],
      sx, sy
    )
  else:
    _ipt2d[uint64_t](
      <uint64_t*>&arrview[0,0],
      sx, sy
    )

  return arr

def ipt3d(cnp.ndarray[COMPLEX_NUMBER, cast=True, ndim=3] arr):
  cdef COMPLEX_NUMBER[:,:,:] arrview = arr

  cdef size_t sx
  cdef size_t sy
  cdef size_t sz

  if arr.flags['F_CONTIGUOUS']:
    sx = arr.shape[0]
    sy = arr.shape[1]
    sz = arr.shape[2]
  else:
    sx = arr.shape[2]
    sy = arr.shape[1]
    sz = arr.shape[0]

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  # ipt doesn't do anything with values, 
  # just moves them around, so only bit width matters
  # int, uint, float, bool who cares
  if nbytes == 1:
    _ipt3d[uint8_t](
      <uint8_t*>&arrview[0,0,0],
      sx, sy, sz
    )
  elif nbytes == 2:
    _ipt3d[uint16_t](
      <uint16_t*>&arrview[0,0,0],
      sx, sy, sz
    )
  elif nbytes == 4:
    _ipt3d[uint32_t](
      <uint32_t*>&arrview[0,0,0],
      sx, sy, sz
    )
  else:
    _ipt3d[uint64_t](
      <uint64_t*>&arrview[0,0,0],
      sx, sy, sz
    )    

  return arr

def ipt4d(cnp.ndarray[COMPLEX_NUMBER, cast=True, ndim=4] arr):
  cdef COMPLEX_NUMBER[:,:,:,:] arrview = arr

  cdef size_t sx
  cdef size_t sy
  cdef size_t sz
  cdef size_t sw

  if arr.flags['F_CONTIGUOUS']:
    sx = arr.shape[0]
    sy = arr.shape[1]
    sz = arr.shape[2]
    sw = arr.shape[3]
  else:
    sx = arr.shape[3]
    sy = arr.shape[2]
    sz = arr.shape[1]
    sw = arr.shape[0]

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  # ipt doesn't do anything with values, 
  # just moves them around, so only bit width matters
  # int, uint, float, bool who cares
  if nbytes == 1:
    _ipt4d[uint8_t](
      <uint8_t*>&arrview[0,0,0,0],
      sx, sy, sz, sw
    )
  elif nbytes == 2:
    _ipt4d[uint16_t](
      <uint16_t*>&arrview[0,0,0,0],
      sx, sy, sz, sw
    )
  elif nbytes == 4:
    _ipt4d[uint32_t](
      <uint32_t*>&arrview[0,0,0,0],
      sx, sy, sz, sw
    )
  else:
    _ipt4d[uint64_t](
      <uint64_t*>&arrview[0,0,0,0],
      sx, sy, sz, sw
    )

  return arr

