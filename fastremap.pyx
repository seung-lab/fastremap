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
Date: August 2018 - October 2019
"""

cimport cython
from libc.stdint cimport (  
  uint8_t, uint16_t, uint32_t, uint64_t,
  int8_t, int16_t, int32_t, int64_t
)
from libcpp.unordered_map cimport unordered_map

from functools import reduce
import operator

import numpy as np
cimport numpy as cnp

__version__ = '1.6.2'

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

ctypedef fused NUMBER:
  ALLINT
  float
  double

ctypedef fused COMPLEX_NUMBER:
  NUMBER
  float complex 

cdef extern from "ipt.hpp" namespace "pyipt":
  cdef void _ipt2d[T](T* arr, int sx, int sy)
  cdef void _ipt3d[T](
    T* arr, int sx, int sy, int sz
  )
  cdef void _ipt4d[T](
    T* arr, int sx, int sy, int sz, int sw
  )

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
  cdef dict remap_dict = {}
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

    if elem in remap_dict:
      arrview[i] = remap_dict[elem]
    else:
      arrview[i] = remap_id
      remap_dict[elem] = remap_id
      remap_id += 1

    last_elem = elem 
    last_remap_id = arrview[i]

  if start < 0:
    types = [ np.int8, np.int16, np.int32, np.int64 ]
  else:
    types = [ np.uint8, np.uint16, np.uint32, np.uint64 ]
  
  factor = max(abs(start), abs(remap_id))

  if factor < 2 ** 8:
    final_type = types[0]
  elif factor < 2 ** 16:
    final_type = types[1]
  elif factor < 2 ** 32:
    final_type = types[2]
  else:
    final_type = types[3]

  if arr.dtype == final_type:
    return arr, remap_dict
  else:
    return arr.astype(final_type), remap_dict

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

  cdef unordered_map[ALLINT, ALLINT] tbl 

  for label in labels:
    tbl[label] = label 

  if value == 0:
    for i in range(size):
      arrview[i] = tbl[arrview[i]]
  else:
    for i in range(size):
      if tbl.find(arrview[i]) == tbl.end():
        arrview[i] = value

  return arr


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
  shape = arr.shape 

  if arr.flags['F_CONTIGUOUS']:
    order = 'F'
  else:
    order = 'C'

  if not in_place:
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

  cdef unordered_map[NUMBER, NUMBER] tbl 

  for k, v in table.items():
    tbl[k] = v 

  for i in range(size):
    elem = arrview[i]
    if tbl.find(elem) == tbl.end():
      if preserve_missing_labels:
        continue
      else:
        raise KeyError("{} was not in the remap table.".format(elem))  
    else:
      arrview[i] = tbl[elem]

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

def transpose(arr):
  """
  asfortranarray(arr)

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

  cdef int sx
  cdef int sy

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

  cdef int sx
  cdef int sy
  cdef int sz

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

  cdef int sx
  cdef int sy
  cdef int sz
  cdef int sw

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

