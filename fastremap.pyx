"""
Functions related to remapping volumes into smaller data types.
For example, a uint64 volume can contain values as high as 2^64,
however, if the volume is only 512x512x512 voxels, the maximum
spread of values would be 134,217,728 (ratio: 7.2e-12). 

For some operations, we can save memory and improve performance
by performing operations on a remapped volume and remembering the
mapping back to the original value.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018 - January 2019
"""

cimport cython
from libc.stdint cimport (  
  uint8_t, uint16_t, uint32_t, uint64_t,
  int8_t, int16_t, int32_t, int64_t
)
from libcpp.unordered_map cimport unordered_map

import numpy as np
cimport numpy as cnp

__version__ = '1.0.1'

ctypedef fused ALLINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  int8_t
  int16_t
  int32_t
  int64_t

ctypedef fused NUMBER:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  int8_t
  int16_t
  int32_t
  int64_t
  float
  double

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

cdef extern from "ipt.hpp" namespace "pyipt":
  cdef void _ipt2d[T](T* arr, int sx, int sy)
  cdef void _ipt3d[T](
    T* arr, int sx, int sy, int sz
  )
  cdef void _ipt4d[T](
    T* arr, int sx, int sy, int sz, int sw
  )

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def renumber(arr, uint64_t start=1, preserve_zero=True):
  """
  renumber(arr, uint64_t start=1, preserve_zero=True)

  Given an array of integers, renumber all the unique values starting
  from 1. This can allow us to reduce the size of the data width required
  to represent it.

  arr: A numpy array
  start (default: 1): Start renumbering from this value
  preserve_zero (default ): 

  Return: a renumbered array, dict with remapping of oldval => newval
  """
  shape = arr.shape

  cdef uint64_t[:] arrview64 
  cdef uint32_t[:] arrview32
  cdef uint16_t[:] arrview16
  cdef uint8_t[:] arrview8

  dtype_bytes = np.dtype(arr.dtype).itemsize
  order = 'F' if arr.flags['F_CONTIGUOUS'] else 'C'

  if dtype_bytes == 8:
    arrview64 = arr.astype(np.uint64).flatten(order)
  elif dtype_bytes == 4:
    arrview32 = arr.astype(np.uint32).flatten(order)
  elif dtype_bytes == 2:
    arrview16 = arr.astype(np.uint16).flatten(order)
  else:
    arrview8 = arr.astype(np.uint8).flatten(order)

  cdef dict remap_dict = {}

  if preserve_zero:
    remap_dict[0] = 0
  
  cdef uint64_t remap_id = start
  cdef int i = 0

  cdef uint64_t elem
  cdef int size = arr.size
  if dtype_bytes == 8:
    for i in range(size):
      elem = arrview64[i]
      if elem in remap_dict:
        arrview64[i] = remap_dict[elem]
      else:
        arrview64[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1
  elif dtype_bytes == 4:
    for i in range(size):
      elem = arrview32[i]
      if elem in remap_dict:
        arrview32[i] = remap_dict[elem]
      else:
        arrview32[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1
  elif dtype_bytes == 2:
    for i in range(size):
      elem = arrview16[i]
      if elem in remap_dict:
        arrview16[i] = remap_dict[elem]
      else:
        arrview16[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1
  else:
    for i in range(size):
      elem = arrview8[i]
      if elem in remap_dict:
        arrview8[i] = remap_dict[elem]
      else:
        arrview8[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1

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

  if dtype_bytes == 8:
    output = bytearray(arrview64)
    intermediate_dtype = np.uint64
  elif dtype_bytes == 4:
    output = bytearray(arrview32)
    intermediate_dtype = np.uint32
  elif dtype_bytes == 2:
    output = bytearray(arrview16)
    intermediate_dtype = np.uint16
  else:
    output = bytearray(arrview8)
    intermediate_dtype = np.uint8

  output = np.frombuffer(output, dtype=intermediate_dtype).astype(final_type)
  output = output.reshape( arr.shape, order=order)
  return output, remap_dict

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)          
cpdef cnp.ndarray[ALLINT] remap(cnp.ndarray[ALLINT] arr, dict table, preserve_missing_labels=False):
  """
  remap(cnp.ndarray[ALLINT] arr, dict table, preserve_missing_labels=False)

  Remap an input numpy array in-place according to the values in the given 
  dictionary "table". Depending on the value of "preserve_missing_labels", 
  if an array value is not present in "table", leave it alone or throw a KeyError.

  Returns: remapped array
  """
  cdef ALLINT[:] arrview = arr
  cdef int i = 0

  cdef int size = arr.size

  for i in range(size):
    elem = arr[i]
    try:
      arrview[i] = table[elem]
    except KeyError:
      if preserve_missing_labels:
        continue
      else:
        raise

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

def asfortranarray(arr):
  """
  For square and cubic matrices, perform in-place transposition. 
  Otherwise default to the out-of-place implementation numpy uses.
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

  if arr.ndim == 2:
    arr = ipt2d(arr)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(nbytes, shape[0] * nbytes))
  elif arr.ndim == 3:
    arr = ipt3d(arr)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(nbytes, shape[0] * nbytes, shape[0] * shape[1] * nbytes))
  elif arr.ndim == 4:
    arr = ipt4d(arr)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, 
      strides=(
        nbytes, 
        shape[0] * nbytes, 
        shape[0] * shape[1] * nbytes, 
        shape[0] * shape[1] * shape[2] * nbytes
      ))
  else:
    return np.asfortranarray(arr)

def ascontiguousarray(arr):
  """
  For square and cubic matrices, perform in-place transposition. 
  Otherwise default to the out-of-place implementation numpy uses.
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

  if arr.ndim == 2:
    arr = ipt2d(arr)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(shape[1] * nbytes, nbytes))
  elif arr.ndim == 3:
    arr = ipt3d(arr)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=(
        shape[2] * shape[1] * nbytes, 
        shape[2] * nbytes, 
        nbytes,
      ))
  elif arr.ndim == 4:
    arr = ipt4d(arr)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, 
      strides=(
        shape[3] * shape[2] * shape[1] * nbytes,
        shape[3] * shape[2] * nbytes, 
        shape[3] * nbytes, 
        nbytes, 
      ))
  else:
    return np.ascontiguousarray(arr)

def ipt2d(cnp.ndarray[NUMBER, cast=True, ndim=2] arr):
  cdef NUMBER[:,:] arrview = arr

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

def ipt3d(cnp.ndarray[NUMBER, cast=True, ndim=3] arr):
  cdef NUMBER[:,:,:] arrview = arr

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

def ipt4d(cnp.ndarray[NUMBER, cast=True, ndim=4] arr):
  cdef NUMBER[:,:,:,:] arrview = arr

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

