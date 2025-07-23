# cython: language_level=3
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
Date: August 2018 - May 2025
"""
from typing import Sequence, List
cimport cython
from libc.stdint cimport (  
  uint8_t, uint16_t, uint32_t, uint64_t,
  int8_t, int16_t, int32_t, int64_t,
  uintptr_t
)
cimport fastremap

from collections import defaultdict
from functools import reduce
import operator

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libcpp.vector cimport vector

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

def minmax(arr):
  """
  Returns (min(arr), max(arr)) computed in a single pass.
  Returns (None, None) if array is size zero.
  """
  return _minmax(_reshape(arr, (arr.size,)))

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

def _match_array_orders(*arrs, order="K"):
  if len(arrs) == 0:
    return []

  if order == "C" or (order == "K" and arrs[0].flags.c_contiguous):
    return [ np.ascontiguousarray(arr) for arr in arrs ]
  else:
    return [ np.asfortranarray(arr) for arr in arrs ]

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def indices(cnp.ndarray[NUMBER, cast=True, ndim=1] arr, NUMBER value):
  """
  Search through an array and identify the indices where value matches the array.
  """
  cdef vector[uint64_t] all_indices
  cdef uint64_t i = 0
  cdef uint64_t size = arr.size

  for i in range(size):
    if arr[i] == value:
      all_indices.push_back(i)

  return np.asarray(all_indices, dtype=np.uint64)

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
  arr = np.asarray(arr)

  if arr.size == 0:
    return arr, {}

  if arr.dtype == bool and preserve_zero:
    return arr, { 0: 0, 1: start }
  elif arr.dtype == bool:
    arr = arr.view(np.uint8)

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  shape = arr.shape
  order = 'F' if arr.flags['F_CONTIGUOUS'] else 'C'
  in_place = in_place and (arr.flags['F_CONTIGUOUS'] or arr.flags['C_CONTIGUOUS'])

  if not in_place:
    arr = np.copy(arr, order=order)

  arr = np.lib.stride_tricks.as_strided(arr, shape=(arr.size,), strides=(nbytes,))
  arr, remap_dict = _renumber(arr, <int64_t>start, preserve_zero)
  arr = _reshape(arr, shape, order)

  return arr, remap_dict

def _reshape(arr, shape, order=None):
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
  cdef flat_hash_map[NUMBER, NUMBER] remap_dict

  if arr.size == 0:
    return refit(np.zeros((0,), dtype=arr.dtype), 0), {}

  remap_dict.reserve(1024)

  if preserve_zero:
    remap_dict[0] = 0

  cdef NUMBER[:] arrview = arr

  cdef NUMBER remap_id = start
  cdef NUMBER elem

  # some value that isn't the first value
  # and won't cause an overflow
  cdef NUMBER last_elem = <NUMBER>(~<uint64_t>arr[0])
  cdef NUMBER last_remap_id = start

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

  return refit(arr, factor), { k:v for k,v in remap_dict }

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

  Return: refitted dtype
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

def widen_dtype(dtype, exotics:bool = False):
  """
  Widen the given dtype to the next size
  of the same type. For example, 
  int8 -> int16 or uint32 -> uint64

  64-bit types will map to themselves.

  Return: upgraded dtype
  """
  dtype = np.dtype(dtype)

  if np.issubdtype(dtype, np.floating):
    sequence = [ np.float16, np.float32, np.float64 ] 
    if exotics:
      sequence += [ np.longdouble ]
  elif np.issubdtype(dtype, np.unsignedinteger):
    sequence = [ np.uint8, np.uint16, np.uint32, np.uint64 ]
  elif np.issubdtype(dtype, np.complexfloating):
    sequence = [ np.complex64 ]
    if exotics:
      sequence += [ np.complex128, np.clongdouble ]
  elif np.issubdtype(dtype, np.integer):
    sequence = [ np.int8, np.int16, np.int32, np.int64 ]
  elif np.issubdtype(dtype, (np.intp, np.uintp)):
    return dtype
  elif exotics:
    raise ValueError(
      f"Unsupported dtype: {dtype}\n"
    )    
  else:
    raise ValueError(
      f"Unsupported dtype: {dtype}\n"
      f"Only standard floats, integers, and complex types are supported."
      f"For additional types (e.g. long double, complex128, clongdouble), enable exotics."
    )

  idx = sequence.index(dtype)
  return sequence[min(idx+1, len(sequence) - 1)]

def narrow_dtype(dtype, exotics:bool = False):
  """
  Widen the given dtype to the next size
  of the same type. For example, 
  int16 -> int8 or uint64 -> uint32

  8-bit types will map to themselves.

  exotics: include float16

  Return: upgraded dtype
  """
  dtype = np.dtype(dtype)
  if dtype.itemsize == 1:
    return dtype

  if np.issubdtype(dtype, np.floating):
    sequence = [ np.float32, np.float64, np.longdouble ] 
    if exotics:
      sequence = [ np.float16 ] + sequence
  elif np.issubdtype(dtype, np.unsignedinteger):
    sequence = [ np.uint8, np.uint16, np.uint32, np.uint64 ]
  elif np.issubdtype(dtype, np.complexfloating):
    sequence = [ np.complex64, np.complex128, np.clongdouble ]
  elif np.issubdtype(dtype, np.integer):
    sequence = [ np.int8, np.int16, np.int32, np.int64 ]
  elif np.issubdtype(dtype, (np.intp, np.uintp)):
    return dtype
  else:
    raise ValueError(
      f"Unsupported dtype: {dtype}\n"
      f"Only standard floats, integers, and complex types are supported."
    )

  idx = sequence.index(dtype)
  return sequence[max(idx-1, 0)]

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

  arr = _reshape(arr, (arr.size,))
  arr = _mask_except(arr, labels, value)
  return _reshape(arr, shape, order=order)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)          
def _mask_except(cnp.ndarray[ALLINT] arr, list labels, ALLINT value):
  cdef ALLINT[:] arrview = arr
  cdef size_t i = 0
  cdef size_t size = arr.size

  if size == 0:
    return arr

  cdef flat_hash_map[ALLINT, ALLINT] tbl 

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

  component_labels, parent_labels = _match_array_orders(
    component_labels, parent_labels
  )

  component_labels = _reshape(component_labels, (component_labels.size,))
  parent_labels = _reshape(parent_labels, (parent_labels.size,))
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
  component_labels, parent_labels = _match_array_orders(
    component_labels, parent_labels
  )
  component_labels = _reshape(component_labels, (component_labels.size,))
  parent_labels = _reshape(parent_labels, (parent_labels.size,))
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

  remap = defaultdict(set)
  cdef size_t i = 0

  cdef ALLINT last_label = parent_labels[0]
  cdef ALLINT_2 last_component = component_labels[0]
  remap[parent_labels[0]].add(component_labels[0])
  for i in range(size):
    if last_label == parent_labels[i] and last_component == component_labels[i]:
      continue
    remap[parent_labels[i]].add(component_labels[i])
    last_label = parent_labels[i]
    last_component = component_labels[i]

  # for backwards compatibility
  for key in remap:
    remap[key] = list(remap[key])
  remap.default_factory = list

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

  if all([ k == v for k,v in table.items() ]) and preserve_missing_labels:
    return arr

  arr = _reshape(arr, (arr.size,))
  arr = _remap(arr, table, preserve_missing_labels)
  return _reshape(arr, shape, order=order)

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

  # fast path for remapping only a single label
  # e.g. for masking something out
  cdef NUMBER before = 0
  cdef NUMBER after = 0
  if preserve_missing_labels and len(table) == 1:
    before = next(iter(table.keys()))
    after = table[before]
    if before == after:
      return arr
    for i in range(size):
      if arr[i] == before:
        arr[i] = after
    return arr
    
  cdef flat_hash_map[NUMBER, NUMBER] tbl 

  for k, v in table.items():
    tbl[k] = v 

  cdef NUMBER last_elem = arrview[0]
  cdef NUMBER last_remap_id = 0
  
  with nogil:
    if tbl.find(last_elem) == tbl.end():
      if not preserve_missing_labels:
        raise KeyError("{} was not in the remap table.".format(last_elem))
      else:
        last_remap_id = last_elem
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
def remap_from_array(cnp.ndarray[UINT] arr, cnp.ndarray[UINT] vals, in_place=True):
  """
  remap_from_array(cnp.ndarray[UINT] arr, cnp.ndarray[UINT] vals)
  """
  cdef size_t i = 0
  cdef size_t size = arr.size 
  cdef size_t maxkey = vals.size - 1
  cdef UINT elem

  if not in_place:
    arr = np.copy(arr)

  with nogil:
    for i in range(size):
      elem = arr[i]
      if elem < 0 or elem > maxkey:
        continue
      arr[i] = vals[elem]

  return arr

@cython.boundscheck(False)
def remap_from_array_kv(cnp.ndarray[ALLINT] arr, cnp.ndarray[ALLINT] keys, cnp.ndarray[ALLINT] vals, bint preserve_missing_labels=True, in_place=True):
  """
  remap_from_array_kv(cnp.ndarray[ALLINT] arr, cnp.ndarray[ALLINT] keys, cnp.ndarray[ALLINT] vals)
  """
  cdef flat_hash_map[ALLINT, ALLINT] remap_dict

  assert keys.size == vals.size

  cdef size_t i = 0
  cdef size_t size = keys.size 
  cdef ALLINT elem

  if not in_place:
    arr = np.copy(arr)

  with nogil:
    for i in range(size):
      remap_dict[keys[i]] = vals[i]

  i = 0
  size = arr.size 

  with nogil:
    for i in range(size):
      elem = arr[i]
      if remap_dict.find(elem) == remap_dict.end():
        if preserve_missing_labels:
          continue
        else:
          raise KeyError("{} was not in the remap keys.".format(elem))
      else:
          arr[i] = remap_dict[elem]

  return arr

def pixel_pairs(labels):
  """
  Computes the number of matching adjacent memory locations.

  This is useful for rapidly evaluating whether an image is
  more binary or more connectomics like.
  """
  if labels.size == 0:
    return 0
  return _pixel_pairs(_reshape(labels, (labels.size,)))

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

@cython.binding(True)
def unique(labels, return_index=False, return_inverse=False, return_counts=False, axis=None):
  """
  Compute the sorted set of unique labels in the input array.

  return_index: also return the index of the first detected occurance 
    of each label.
  return_inverse: If True, also return the indices of the unique array 
    (for the specified axis, if provided) that can be used to reconstruct
    the input array.
  return_counts: also return the unique label frequency as an array.

  Returns: 
    unique ndarray
        The sorted unique values.
    
    unique_indices ndarray, optional
        The indices of the first occurrences of the unique values in the original array. 
        Only provided if return_index is True.
    
    unique_inverse ndarray, optional
        The indices to reconstruct the original array from the unique array. 
        Only provided if return_inverse is True.
    
    unique_counts ndarray, optional
        The number of times each of the unique values comes up in the original array. 
        Only provided if return_counts is True.
  """
  if not isinstance(labels, np.ndarray):
    labels = np.array(labels)

  # These flags are currently unsupported so call uncle and
  # use the standard implementation instead.
  if (axis is not None) or (not np.issubdtype(labels.dtype, np.integer)):
    if (
      axis == 0
      and (
        labels.ndim == 2 
        and labels.shape[1] == 2
        and np.dtype(labels.dtype).itemsize < 8 
        and np.issubdtype(labels.dtype, np.integer)
      )
      and not (return_index or return_inverse or return_counts)
      and labels.flags.c_contiguous
    ):
      return _two_axis_unique(labels)
    else:
      return np.unique(
        labels, 
        return_index=return_index, 
        return_inverse=return_inverse, 
        return_counts=return_counts, 
        axis=axis
      )

  cdef size_t voxels = labels.size

  shape = labels.shape
  fortran_order = labels.flags.f_contiguous
  order = "F" if fortran_order else "C"
  labels_orig = labels
  labels = _reshape(labels, (voxels,))

  max_label = 0
  min_label = 0
  if voxels > 0:
    min_label, max_label = minmax(labels)

  def c_order_index(arr):
    if len(shape) > 1 and fortran_order:
      return np.ravel_multi_index(
        np.unravel_index(arr, shape, order='F'), 
        shape, order='C'
      )
    return arr

  if voxels == 0:
    uniq = np.array([], dtype=labels.dtype)
    counts = np.array([], dtype=np.uint32)
    index = np.array([], dtype=np.uint64)
    inverse = np.array([], dtype=np.uintp)
  elif min_label >= 0 and max_label < int(voxels):
    uniq, index, counts, inverse = _unique_via_array(labels, max_label, return_index=return_index, return_inverse=return_inverse)
  elif (max_label - min_label) <= int(voxels):
    uniq, index, counts, inverse = _unique_via_shifted_array(labels, min_label, max_label, return_index=return_index, return_inverse=return_inverse)
  elif float(pixel_pairs(labels)) / float(voxels) > 0.66:
    uniq, index, counts, inverse = _unique_via_renumber(labels, return_index=return_index, return_inverse=return_inverse)
  elif return_index or return_inverse:
    return np.unique(labels_orig, return_index=return_index, return_counts=return_counts, return_inverse=return_inverse)
  else:
    uniq, counts = _unique_via_sort(labels)
    index = None
    inverse = None

  results = [ uniq ]
  if return_index:
    # This is required to match numpy's behavior
    results.append(c_order_index(index))
  if return_inverse:
    results.append(_reshape(inverse, shape, order=order))
  if return_counts:
    results.append(counts)

  if len(results) > 1:
    return tuple(results)
  return uniq

def _two_axis_unique(labels):
  """
  Faster replacement for np.unique(labels, axis=0)
  when ndim = 2 and the dtype can be widened.

  This special case is useful for sorting edge lists.
  """
  dtype = labels.dtype
  wide_dtype = widen_dtype(dtype)

  labels = labels[:, [1,0]].reshape(-1, order="C")
  labels = labels.view(wide_dtype)
  labels = unique(labels)
  N = len(labels)
  labels = labels.view(dtype).reshape((N, 2), order="C")
  return labels[:,[1,0]]

def _unique_via_shifted_array(labels, min_label=None, max_label=None, return_index=False, return_inverse=False):
  if min_label is None or max_label is None:
    min_label, max_label = minmax(labels)

  if labels.flags.writeable:
    labels -= min_label
    arr = labels
  else:
    arr = labels - min_label

  uniq, idx, counts, inverse = _unique_via_array(arr, max_label - min_label + 1, return_index, return_inverse)
  del arr

  if labels.flags.writeable:
    labels += min_label

  uniq += min_label
  return uniq, idx, counts, inverse

def _unique_via_renumber(labels, return_index=False, return_inverse=False):
  dtype = labels.dtype
  labels, remap = renumber(labels)
  remap = { v:k for k,v in remap.items() }
  uniq, idx, counts, inverse = _unique_via_array(labels, max(remap.keys()), return_index, return_inverse)
  uniq = np.array([ remap[segid] for segid in uniq ], dtype=dtype)

  if not return_index and not return_inverse:
    uniq.sort()
    return uniq, idx, counts, inverse

  uniq, idx2 = np.unique(uniq, return_index=return_index)
  if idx is not None:
    idx = idx[idx2]
  if counts is not None:
    counts = counts[idx2]
  if inverse is not None:
    inverse = idx2[inverse]

  return uniq, idx, counts, inverse

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _unique_via_sort(cnp.ndarray[ALLINT, ndim=1] labels):
  """Slower than _unique_via_array but can handle any label."""
  labels = np.copy(labels)
  labels.sort()

  cdef size_t voxels = labels.size  

  cdef vector[ALLINT] uniq
  uniq.reserve(100)

  cdef vector[uint64_t] counts
  counts.reserve(100)

  cdef size_t i = 0

  cdef ALLINT cur = labels[0]
  cdef uint64_t accum = 1
  for i in range(1, voxels):
    if cur == labels[i]:
      accum += 1
    else:
      uniq.push_back(cur)
      counts.push_back(accum)
      accum = 1
      cur = labels[i]

  uniq.push_back(cur)
  counts.push_back(accum)

  dtype = labels.dtype
  del labels

  return np.array(uniq, dtype=dtype), np.array(counts, dtype=np.uint64)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _unique_via_array(
  cnp.ndarray[ALLINT, ndim=1] labels, 
  size_t max_label, 
  return_index, return_inverse,
):
  cdef cnp.ndarray[uint64_t, ndim=1] counts = np.zeros( 
    (max_label+1,), dtype=np.uint64
  )
  cdef cnp.ndarray[uintptr_t, ndim=1] index
  
  cdef uintptr_t sentinel = np.iinfo(np.uintp).max
  if return_index:
    index = np.full( 
      (max_label+1,), sentinel, dtype=np.uintp
    )

  cdef size_t voxels = labels.shape[0]
  cdef size_t i = 0
  for i in range(voxels):
    counts[labels[i]] += 1

  if return_index:
    for i in range(voxels):
      if index[labels[i]] == sentinel:
        index[labels[i]] = i

  cdef size_t real_size = 0
  for i in range(max_label + 1):
    if counts[i] > 0:
      real_size += 1

  cdef cnp.ndarray[ALLINT, ndim=1] segids = np.zeros( 
    (real_size,), dtype=labels.dtype
  )
  cdef cnp.ndarray[uint64_t, ndim=1] cts = np.zeros( 
    (real_size,), dtype=np.uint64
  )
  cdef cnp.ndarray[uintptr_t, ndim=1] idx

  cdef size_t j = 0
  for i in range(max_label + 1):
    if counts[i] > 0:
      segids[j] = i
      cts[j] = counts[i]
      j += 1

  if return_index:
    idx = np.zeros( (real_size,), dtype=np.uintp)
    j = 0
    for i in range(max_label + 1):
      if counts[i] > 0:
        idx[j] = index[i]
        j += 1
  
  cdef cnp.ndarray[uintptr_t, ndim=1] mapping

  if return_inverse:
    if segids.size:
      mapping = np.zeros([segids[segids.size - 1] + 1], dtype=np.uintp)
      for i in range(real_size):
        mapping[segids[i]] = i
      inverse_idx = mapping[labels]
    else:
      inverse_idx = np.zeros([0], dtype=np.uintp)

  ret = [ segids, None, cts, None ]
  if return_index:
    ret[1] = idx
  if return_inverse:
    ret[3] = inverse_idx

  return ret

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
  if arr.dtype == bool:
    arr = arr.view(np.uint8)

  if arr.ndim == 2:
    arr = _internal_ipt2d(arr)
    return arr.view(dtype)
  elif arr.ndim == 3:
    arr = _internal_ipt3d(arr)
    return arr.view(dtype)
  elif arr.ndim == 4:
    arr = _internal_ipt4d(arr)
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
  if arr.dtype == bool:
    arr = arr.view(np.uint8)

  if arr.ndim == 2:
    arr = _internal_ipt2d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(nbytes, shape[0] * nbytes))
    return arr.view(dtype)
  elif arr.ndim == 3:
    arr = _internal_ipt3d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(nbytes, shape[0] * nbytes, shape[0] * shape[1] * nbytes))
    return arr.view(dtype)
  elif arr.ndim == 4:
    arr = _internal_ipt4d(arr)
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
  if arr.dtype == bool:
    arr = arr.view(np.uint8)

  if arr.ndim == 2:
    arr = _internal_ipt2d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(shape[1] * nbytes, nbytes))
    return arr.view(dtype)
  elif arr.ndim == 3:
    arr = _internal_ipt3d(arr)
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(
        shape[2] * shape[1] * nbytes, 
        shape[2] * nbytes, 
        nbytes,
      ))
    return arr.view(dtype)
  elif arr.ndim == 4:
    arr = _internal_ipt4d(arr)
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

def _internal_ipt2d(cnp.ndarray[COMPLEX_NUMBER, cast=True, ndim=2] arr):
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

def _internal_ipt3d(cnp.ndarray[COMPLEX_NUMBER, cast=True, ndim=3] arr):
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

def _internal_ipt4d(cnp.ndarray[COMPLEX_NUMBER, cast=True, ndim=4] arr):
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

def foreground(arr):
  """Returns the number of non-zero voxels in an array."""
  arr = _reshape(arr, (arr.size,))
  return _foreground(arr)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _foreground(cnp.ndarray[ALLINT, ndim=1] arr):
  cdef size_t i = 0
  cdef size_t sz = arr.size
  cdef size_t n_foreground = 0
  for i in range(sz):
    n_foreground += <size_t>(arr[i] != 0)
  return n_foreground

def point_cloud(arr):
  """
  point_cloud(arr)

  Given a 2D or 3D integer image, return a mapping from
  labels to their (x,y,z) position in the image.

  Zero is considered a background label.

  Returns: ndarray(N, 2 or 3, dtype=uint16)
  """
  if arr.dtype == bool:
    arr = arr.view(np.uint8)

  if arr.ndim == 2:
    return _point_cloud_2d(arr)
  else:
    return _point_cloud_3d(arr)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _point_cloud_2d(cnp.ndarray[ALLINT, ndim=2] arr):
  cdef size_t n_foreground = foreground(arr)

  cdef size_t sx = arr.shape[0]
  cdef size_t sy = arr.shape[1]

  if n_foreground == 0:
    return {}

  cdef cnp.ndarray[ALLINT, ndim=1] ptlabel = np.zeros((n_foreground,), dtype=arr.dtype)
  cdef cnp.ndarray[uint16_t, ndim=2] ptcloud = np.zeros((n_foreground, 2), dtype=np.uint16)

  cdef size_t i = 0
  cdef size_t j = 0
  
  cdef size_t idx = 0
  for i in range(sx):
    for j in range(sy):
        if arr[i,j] != 0:
          ptlabel[idx] = arr[i,j]
          ptcloud[idx,0] = i
          ptcloud[idx,1] = j
          idx += 1

  sortidx = ptlabel.argsort()
  ptlabel = ptlabel[sortidx]
  ptcloud = ptcloud[sortidx]
  del sortidx

  ptcloud_by_label = {}
  if n_foreground == 1:
    ptcloud_by_label[ptlabel[0]] = ptcloud
    return ptcloud_by_label

  cdef size_t start = 0
  cdef size_t end = 0
  for end in range(1, n_foreground):
    if ptlabel[end] != ptlabel[end - 1]:
      ptcloud_by_label[ptlabel[end - 1]] = ptcloud[start:end,:]
      start = end

  ptcloud_by_label[ptlabel[end]] = ptcloud[start:,:]

  return ptcloud_by_label

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def _point_cloud_3d(cnp.ndarray[ALLINT, ndim=3] arr):
  cdef size_t n_foreground = foreground(arr)

  cdef size_t sx = arr.shape[0]
  cdef size_t sy = arr.shape[1]
  cdef size_t sz = arr.shape[2]

  if n_foreground == 0:
    return {}

  cdef cnp.ndarray[ALLINT, ndim=1] ptlabel = np.zeros((n_foreground,), dtype=arr.dtype)
  cdef cnp.ndarray[uint16_t, ndim=2] ptcloud = np.zeros((n_foreground, 3), dtype=np.uint16)

  cdef size_t i = 0
  cdef size_t j = 0
  cdef size_t k = 0
  
  cdef size_t idx = 0
  for i in range(sx):
    for j in range(sy):
      for k in range(sz):
        if arr[i,j,k] != 0:
          ptlabel[idx] = arr[i,j,k]
          ptcloud[idx,0] = i
          ptcloud[idx,1] = j
          ptcloud[idx,2] = k
          idx += 1

  sortidx = ptlabel.argsort()
  ptlabel = ptlabel[sortidx]
  ptcloud = ptcloud[sortidx]
  del sortidx

  ptcloud_by_label = {}
  if n_foreground == 1:
    ptcloud_by_label[ptlabel[0]] = ptcloud
    return ptcloud_by_label

  cdef size_t start = 0
  cdef size_t end = 0
  for end in range(1, n_foreground):
    if ptlabel[end] != ptlabel[end - 1]:
      ptcloud_by_label[ptlabel[end - 1]] = ptcloud[start:end,:]
      start = end

  ptcloud_by_label[ptlabel[end]] = ptcloud[start:,:]

  return ptcloud_by_label


@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def tobytes(
  cnp.ndarray[NUMBER, ndim=3] image, 
  chunk_size:Sequence[int,int,int], 
  order:str="C"
) -> List[bytes]:
  """
  Compute the cutout.tobytes(order) with the image divided into 
  a grid of cutouts. Return the resultant binaries indexed by 
  their cutout's gridpoint in fortran order.

  This is faster than calling tobytes on each cutout individually
  if the input and output orders match.
  """
  if order not in ["C", "F"]:
    raise ValueError(f"order must be C or F. Got: {order}")

  chunk_size = np.array(chunk_size, dtype=float)
  shape = np.array((image.shape[0], image.shape[1], image.shape[2]), dtype=float)
  grid_size = np.ceil(shape / chunk_size).astype(int)

  if np.any(np.remainder(shape, chunk_size)):
    raise ValueError(f"chunk_size ({chunk_size}) must evenly divide the image shape ({shape}).")

  chunk_array_size = int(reduce(operator.mul, chunk_size))
  chunk_size = chunk_size.astype(int)
  shape = shape.astype(int)

  num_grid = int(reduce(operator.mul, grid_size))

  cdef int64_t img_i = 0

  cdef int64_t sgx = grid_size[0]
  cdef int64_t sgy = grid_size[1]
  cdef int64_t sgz = grid_size[2]

  cdef int64_t sx = shape[0]
  cdef int64_t sy = shape[1]
  cdef int64_t sz = shape[2]
  cdef int64_t sxy = sx * sy

  cdef int64_t cx = chunk_size[0]
  cdef int64_t cy = chunk_size[1]
  cdef int64_t cz = chunk_size[2]

  cdef int64_t gx = 0
  cdef int64_t gy = 0
  cdef int64_t gz = 0
  cdef int64_t gi = 0

  cdef int64_t idx = 0
  cdef int64_t x = 0
  cdef int64_t y = 0
  cdef int64_t z = 0

  # It's difficult to do better than numpy when f and c or c and f
  # because at least one of the arrays must be transversed substantially
  # out of order. However, when f and f or c and c you can do strips in 
  # order.
  if (
    (not image.flags.f_contiguous and not image.flags.c_contiguous)
    or (image.flags.f_contiguous and order == "C")
    or (image.flags.c_contiguous and order == "F")
  ):
    res = []
    for gz in range(sgz):
      for gy in range(sgy):
        for gx in range(sgx):
          cutout = image[gx*cx:(gx+1)*cx, gy*cy:(gy+1)*cy, gz*cz:(gz+1)*cz]
          res.append(cutout.tobytes(order))
    return res
  elif (cx == sx and cy == sy and cz == sz):
    return [ image.tobytes(order) ]

  cdef cnp.ndarray[NUMBER] arr

  cdef list[cnp.ndarray[NUMBER]] array_grid = [ 
    np.zeros((chunk_array_size,), dtype=image.dtype)
    for i in range(num_grid)
  ]

  cdef cnp.ndarray[NUMBER, ndim=1] img = _reshape(image, (image.size,))

  if order == "F": # b/c of guard above, this is F to F order
    for gz in range(sgz):
      for z in range(cz):
        for gy in range(sgy):
          for gx in range(sgx):
            gi = gx + sgx * (gy + sgy * gz)
            arr = array_grid[gi]
            for y in range(cy):
              img_i = cx * gx + sx * ((cy * gy + y) + sy * (cz * gz + z))
              idx = cx * (y + cy * z)
              for x in range(cx):
                arr[idx + x] = img[img_i + x]
  else: # b/c of guard above, this is C to C order
    for gx in range(sgx):
      for x in range(cx):
        for gy in range(sgy):
          for gz in range(sgz):
            gi = gx + sgx * (gy + sgy * gz)
            arr = array_grid[gi]
            for y in range(cy):
              img_i = cz * gz + sz * ((cy * gy + y) + sy * (cx * gx + x))
              idx = cz * (y + cy * x)
              for z in range(cz):
                arr[idx + z] = img[img_i + z]

  return [ bytes(memoryview(ar)) for ar in array_grid ]
