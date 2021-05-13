import pytest

import numpy as np

import fastremap

DTYPES = (
  np.uint8, np.uint16, np.uint32, np.uint64,
  np.int8, np.int16, np.int32, np.int64
)

def test_empty_renumber():
  for dtype in DTYPES:
    data = np.array([], dtype=dtype)
    data2, remapdict = fastremap.renumber(data, preserve_zero=False)

    assert np.all(data2 == [])
    assert remapdict == {}

def test_1d_renumber():
  for dtype in DTYPES:
    print(dtype)
    data = np.arange(8).astype(dtype)
    data = np.flip(data)

    data2 = np.copy(data)
    data2, remapdict = fastremap.renumber(data2, preserve_zero=False)

    assert np.all(data2 == np.arange(1,9))
    assert len(remapdict) > 0

    data2 = np.copy(data)
    data2, remapdict = fastremap.renumber(data2, preserve_zero=True)

    assert data2[-1] == 0
    assert np.all(data2 == [1,2,3,4,5,6,7,0])
    assert len(remapdict) > 0

  data = np.arange(8).astype(bool)
  data = np.flip(data)

  data2 = np.copy(data)
  data2, remapdict = fastremap.renumber(data2, preserve_zero=False)

  assert np.all(data2 == [1,1,1,1,1,1,1,2])
  assert len(remapdict) > 0

  data2 = np.copy(data)
  data2, remapdict = fastremap.renumber(data2, preserve_zero=True)

  assert np.all(data2 == [1,1,1,1,1,1,1,0])
  assert len(remapdict) > 0

def test_2d_renumber():
  for dtype in DTYPES:
    data = np.array([
      [ 5,  5,  5, 2],
      [ 3,  5,  5, 0],
      [ 1,  2,  4, 1],
      [20, 19, 20, 1],
    ], dtype=dtype)

    data2 = np.copy(data, order='C')
    data2, remapdict = fastremap.renumber(data2, preserve_zero=True)

    assert np.all(data2 == [
      [1, 1, 1, 2],
      [3, 1, 1, 0],
      [4, 2, 5, 4],
      [6, 7, 6, 4],
    ])

    data2 = np.copy(data, order='F')
    data2, remapdict = fastremap.renumber(data2, preserve_zero=True)

    assert np.all(data2 == [
      [1, 1, 1, 5],
      [2, 1, 1, 0],
      [3, 5, 7, 3],
      [4, 6, 4, 3],
    ])

@pytest.mark.parametrize("dtype", DTYPES)
def test_3d_renumber(dtype):
  bits = np.dtype(dtype).itemsize * 8
  big = (2 ** (bits - 1)) - 1 # cover ints and uints
  data = np.array([
    [
      [big, 0],
      [2, big],
    ],
    [
      [big-5, big-1],
      [big-7, big-3],
    ],
  ], dtype=dtype)

  data2 = np.copy(data, order='C')
  data2, remapdict = fastremap.renumber(data2, preserve_zero=False)

  assert np.all(data2 == [
    [
      [1, 2],
      [3, 1]
    ],
    [ 
      [4, 5],
      [6, 7],
    ],
  ])

  data2 = np.copy(data, order='F')
  data2, remapdict = fastremap.renumber(data2, preserve_zero=False)

  assert np.all(data2 == [
    [
      [1, 5],
      [3, 1]
    ],
    [ 
      [2, 6],
      [4, 7],
    ],
  ])

def test_3d_renumber_dtype_shift():
  big = np.random.randint(0, (2**64)-1, size=(128,128,100), dtype=np.uint64)
  big, remapdict = fastremap.renumber(big, preserve_zero=True, in_place=True)
  assert np.dtype(big.dtype).itemsize <= 4
  assert np.dtype(big.dtype).itemsize > 1

def test_renumber_no_preserve_zero():
  data = np.array([
    [0, 1],
    [1, 2]
  ])
  gt = np.array([
    [0, 1],
    [1, 2]
  ])

  res, remap = fastremap.renumber(data, start=0, preserve_zero=False)
  print(res)
  print(remap)
  assert remap == { 0: 0, 1: 1, 2: 2 }
  assert np.all(res == gt)


@pytest.mark.parametrize("dtype", list(DTYPES) + [ np.float32, np.float64 ])
def test_remap_1d(dtype):
  empty = fastremap.remap([], {})
  assert len(empty) == 0

  data = np.array([1, 2, 2, 2, 3, 4, 5], dtype=dtype)
  remap = {
    1: 10,
    2: 30,
    3: 15,
    4: 0,
    5: 5,
  }

  result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=False)
  assert np.all(result == [10, 30, 30, 30, 15, 0, 5])

  del remap[2]
  try:
    result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=False)
    assert False
  except KeyError:
    pass 

  result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=True)
  assert np.all(result == [10, 2, 2, 2, 15, 0, 5])

@pytest.mark.parametrize("dtype", DTYPES)
def test_remap_2d(dtype):
  data = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=dtype)
  remap = {
    1: 10,
    2: 30,
    3: 15,
    4: 0,
    5: 5,
  }

  result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=False)
  assert np.all(result == [[10, 30, 15, 0, 5], [5, 0, 15, 30, 10]])

  del remap[2]
  try:
    result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=False)
    assert False
  except KeyError:
    pass 

  result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=True)
  assert np.all(result == [[10, 2, 15, 0, 5], [5, 0, 15, 2, 10]])

def test_remap_broken():
  labels = np.zeros((256, 256, 256), dtype=np.uint32)

  labels[:50, :40, :30] = 2
  labels[50:100, 40:100, 30:80] = 5

  res = fastremap.remap(labels, {5:5}, preserve_missing_labels=True)
  assert np.all(res == labels)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [ True, False ])
def test_mask(dtype, in_place):
  data = np.arange(100, dtype=dtype)
  data = fastremap.mask(data, [5, 10, 15, 20], in_place=in_place)

  labels, cts = np.unique(data, return_counts=True)
  assert cts[0] == 5 
  assert labels[0] == 0
  assert np.all(cts[1:] == 1)
  assert len(labels == 95)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [ True, False ])
def test_mask_except(dtype, in_place):
  for value in (0, 7, np.iinfo(dtype).max):
    data = np.arange(100, dtype=dtype)
    data = fastremap.mask_except(
      data, [5, 10, 15, 20], 
      in_place=in_place, value=value
    )

    labels, cts = np.unique(data, return_counts=True)
    print(labels, cts)
    res = { lbl: ct for lbl, ct in zip(labels, cts) }
    assert res == {
      value: 96,
      5: 1,
      10: 1,
      15: 1, 
      20: 1,
    }

@pytest.mark.parametrize("dtype", list(DTYPES) + [ np.float32, np.float64, bool, np.complex64 ])
@pytest.mark.parametrize("dim", [1, 4, 7, 9, 27, 31, 100, 127, 200] )
def test_asfortranarray(dtype, dim):
  x = np.arange(dim**1).reshape((dim)).astype(dtype)
  y = np.copy(x)
  assert np.all(np.asfortranarray(x) == fastremap.asfortranarray(y))

  x = np.arange(dim**2).reshape((dim,dim)).astype(dtype)
  y = np.copy(x)
  assert np.all(np.asfortranarray(x) == fastremap.asfortranarray(y))

  x = np.arange(dim**3).reshape((dim,dim,dim)).astype(dtype)
  y = np.copy(x)
  assert np.all(np.asfortranarray(x) == fastremap.asfortranarray(y))

  x = np.arange(dim**2+dim).reshape((dim,dim+1)).astype(dtype)
  y = np.copy(x)
  assert np.all(np.asfortranarray(x) == fastremap.asfortranarray(y))

  x = np.arange(dim**3+dim*dim).reshape((dim,dim+1,dim)).astype(dtype)
  y = np.copy(x)
  assert np.all(np.asfortranarray(x) == fastremap.asfortranarray(y))

  if dim < 100:
    x = np.arange(dim**4).reshape((dim,dim,dim,dim)).astype(dtype)
    y = np.copy(x)
    assert np.all(np.asfortranarray(x) == fastremap.asfortranarray(y))

    x = np.arange(dim**4 + dim*dim*dim).reshape((dim+1,dim,dim,dim)).astype(dtype)
    y = np.copy(x)
    assert np.all(np.asfortranarray(x) == fastremap.asfortranarray(y))


@pytest.mark.parametrize("dtype", list(DTYPES) + [ np.float32, np.float64, bool, np.complex64 ])
@pytest.mark.parametrize("dim", [1, 4, 7, 9, 27, 31, 100, 127, 200] )
def test_ascontiguousarray(dtype, dim):
  x = np.arange(dim**2).reshape((dim,dim), order='F').astype(dtype)
  y = np.copy(x, order='F')
  assert np.all(np.ascontiguousarray(x) == fastremap.ascontiguousarray(y))

  x = np.arange(dim**3).reshape((dim,dim,dim), order='F').astype(dtype)
  y = np.copy(x, order='F')
  assert np.all(np.ascontiguousarray(x) == fastremap.ascontiguousarray(y))

  x = np.arange(dim**2+dim).reshape((dim,dim+1), order='F').astype(dtype)
  y = np.copy(x, order='F')
  assert np.all(np.ascontiguousarray(x) == fastremap.ascontiguousarray(y))

  x = np.arange(dim**3+dim*dim).reshape((dim,dim+1,dim), order='F').astype(dtype)
  y = np.copy(x, order='F')
  assert np.all(np.ascontiguousarray(x) == fastremap.ascontiguousarray(y))

  if dim < 100:
    x = np.arange(dim**4).reshape((dim,dim,dim,dim)).astype(dtype)
    y = np.copy(x, order='F')
    assert np.all(np.ascontiguousarray(x) == fastremap.ascontiguousarray(y))

    x = np.arange(dim**4 + dim*dim*dim).reshape((dim+1,dim,dim,dim)).astype(dtype)
    y = np.copy(x, order='F')
    assert np.all(np.ascontiguousarray(x) == fastremap.ascontiguousarray(y))

@pytest.mark.parametrize("dtype", [ np.uint8, np.uint16, np.uint32, np.uint64 ])
def test_fit_dtype_uint(dtype):
  assert fastremap.fit_dtype(dtype, 0) == np.uint8
  assert fastremap.fit_dtype(dtype, 255) == np.uint8
  assert fastremap.fit_dtype(dtype, 256) == np.uint16
  assert fastremap.fit_dtype(dtype, 10000) == np.uint16
  assert fastremap.fit_dtype(dtype, 2**16 - 1) == np.uint16
  assert fastremap.fit_dtype(dtype, 2**16) == np.uint32
  assert fastremap.fit_dtype(dtype, 2**32) == np.uint64
  assert fastremap.fit_dtype(dtype, 2**64 - 1) == np.uint64

  try:
    fastremap.fit_dtype(dtype, -1)
    assert False 
  except ValueError:
    pass

  try:
    fastremap.fit_dtype(dtype, 2**64)
  except ValueError:
    pass

@pytest.mark.parametrize("dtype", [ np.int8, np.int16, np.int32, np.int64 ])
def test_fit_dtype_int(dtype):
  assert fastremap.fit_dtype(dtype, 0) == np.int8
  assert fastremap.fit_dtype(dtype, 127) == np.int8
  assert fastremap.fit_dtype(dtype, -128) == np.int8
  assert fastremap.fit_dtype(dtype, 128) == np.int16
  assert fastremap.fit_dtype(dtype, 10000) == np.int16
  assert fastremap.fit_dtype(dtype, 2**15 - 1) == np.int16
  assert fastremap.fit_dtype(dtype, 2**15) == np.int32
  assert fastremap.fit_dtype(dtype, 2**32) == np.int64
  assert fastremap.fit_dtype(dtype, 2**63 - 1) == np.int64

  try:
    fastremap.fit_dtype(dtype, 2**63)
  except ValueError:
    pass

  try:
    fastremap.fit_dtype(dtype, -2**63)
  except ValueError:
    pass

@pytest.mark.parametrize("dtype", [ np.float16, np.float32, np.float64 ])
def test_fit_dtype_float(dtype):
  assert fastremap.fit_dtype(dtype, 0) == np.float32
  assert fastremap.fit_dtype(dtype, 127) == np.float32
  assert fastremap.fit_dtype(dtype, 128) == np.float32
  assert fastremap.fit_dtype(dtype, 10000) == np.float32
  assert fastremap.fit_dtype(dtype, 2**15 - 1) == np.float32
  assert fastremap.fit_dtype(dtype, 2**15) == np.float32
  assert fastremap.fit_dtype(dtype, 2**32) == np.float32
  assert fastremap.fit_dtype(dtype, 2**63 - 1) == np.float32
  assert fastremap.fit_dtype(dtype, -2**63) == np.float32
  assert fastremap.fit_dtype(dtype, 2**128) == np.float64

  assert fastremap.fit_dtype(dtype, 0, exotics=True) == np.float16
  assert fastremap.fit_dtype(dtype, 127, exotics=True) == np.float16
  assert fastremap.fit_dtype(dtype, 128, exotics=True) == np.float16
  assert fastremap.fit_dtype(dtype, 10000, exotics=True) == np.float16
  assert fastremap.fit_dtype(dtype, 2**15 - 1, exotics=True) == np.float16
  assert fastremap.fit_dtype(dtype, 2**15, exotics=True) == np.float16
  assert fastremap.fit_dtype(dtype, 2**32, exotics=True) == np.float32
  assert fastremap.fit_dtype(dtype, 2**63 - 1, exotics=True) == np.float32
  assert fastremap.fit_dtype(dtype, -2**63, exotics=True) == np.float32

@pytest.mark.parametrize("dtype", [ np.csingle, np.cdouble ])
@pytest.mark.parametrize("sign", [ 1, -1, 1j, -1j ])
def test_fit_dtype_float(dtype, sign):
  assert fastremap.fit_dtype(dtype, sign * 0+0j) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 127) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 127) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 128) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 128) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 10000) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 10000) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 2**15 - 1) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 2**15) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 2**32) == np.csingle
  assert fastremap.fit_dtype(dtype, sign * 2**63 - 1) == np.csingle
  assert fastremap.fit_dtype(dtype, -2**63) == np.csingle
  
  try:
    fastremap.fit_dtype(dtype, sign * 2**128)
    assert False
  except ValueError:
    pass

  assert fastremap.fit_dtype(dtype, sign * 2**128, exotics=True) == np.cdouble

def test_minmax():
  volume = np.random.randint(-500, 500, size=(128,128,128))
  minval, maxval = fastremap.minmax(volume)
  assert minval == np.min(volume)
  assert maxval == np.max(volume)

@pytest.mark.parametrize("order", [ "C", "F" ])
def test_unique(order):
  def reorder(arr):
    if order == "F":
      return np.asfortranarray(arr)
    return np.ascontiguousarray(arr)

  assert len(fastremap.unique(np.array([], dtype=np.uint8))) == 0

  # array_unique
  labels = reorder(np.random.randint(0, 500, size=(128,128,128)))
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique(labels, return_counts=True, return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  labels = reorder(np.random.randint(0, 500, size=(128,128,128)))
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True,return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique_via_array(labels.flatten(), np.max(labels), return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  # array_unique + shift
  labels = reorder(np.random.randint(-500, 500, size=(128,128,128)))
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique(labels, return_counts=True, return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  labels = reorder(np.random.randint(-500, 500, size=(128,128,128)))
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique_via_shifted_array(labels.flatten(), return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  # array_unique + shift
  labels = reorder(np.random.randint(128**3 - 500, 128**3 + 500, size=(128,128,128)))
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique(labels, return_counts=True, return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  # array_unique + shift
  labels = reorder(np.random.randint(128**3 - 500, 128**3 + 500, size=(128,128,128)))
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique_via_shifted_array(labels.flatten(), return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  # renumber + array_unique
  labels = reorder(np.random.randint(0, 1, size=(128,128,128)))
  labels[0,0,0] = 128**3 + 10
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique(labels, return_counts=True, return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)  
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  labels = reorder(np.random.randint(0, 1, size=(128,128,128)))
  labels[0,0,0] = 128**3 + 10
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique_via_renumber(labels.flatten(), return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)  
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  # sort
  labels = reorder(np.random.randint(-1000, 128**3, size=(100,100,100)))
  uniq_np, idx_np, cts_np = np.unique(labels, return_counts=True, return_index=True)
  uniq_fr, idx_fr, cts_fr = fastremap.unique(labels, return_counts=True, return_index=True)
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)  
  assert np.all(labels.flatten()[idx_np] == labels.flatten()[idx_fr])

  labels = reorder(np.random.randint(-1000, 128**3, size=(100,100,100)))
  uniq_np, cts_np = np.unique(labels, return_counts=True)
  uniq_fr, cts_fr = fastremap.unique_via_sort(labels.flatten())
  assert np.all(uniq_np == uniq_fr)
  assert np.all(cts_np == cts_fr)

def test_renumber_remap():
  labels = np.random.randint(-500, 500, size=(128,128,128)).astype(np.int64)
  new_labels, remap = fastremap.renumber(labels, in_place=False)
  remap = { v:k for k,v in remap.items() }
  new_labels = fastremap.remap(new_labels, remap, in_place=True)
  assert np.all(labels == new_labels)
  assert new_labels.dtype in (np.int8, np.int16)
  assert labels.dtype == np.int64

@pytest.mark.parametrize("dtype_cc", DTYPES)
@pytest.mark.parametrize("dtype_p", DTYPES)
def test_component_map(dtype_cc, dtype_p):
  shape = (128,128,128)
  cc_labels = np.random.randint(0, 100, size=shape).astype(dtype_cc)
  parent_labels = (cc_labels + 1).astype(dtype_p)

  mapping = fastremap.component_map(cc_labels, parent_labels)
  for k,v in mapping.items():
    assert k == v - 1

  mapping = fastremap.component_map([ 1, 2, 3, 4 ], [ 5, 5, 6, 7 ])
  assert mapping == { 1: 5, 2: 5, 3: 6, 4: 7 }

  mapping = fastremap.component_map([], [])


@pytest.mark.parametrize("dtype_cc", DTYPES)
@pytest.mark.parametrize("dtype_p", DTYPES)
def test_inverse_component_map(dtype_cc, dtype_p):
  mapping = fastremap.inverse_component_map([ 1, 2, 1, 3 ], [ 4, 4, 5, 6 ])
  assert mapping == { 1: [ 4, 5 ], 2: [ 4 ], 3: [ 6 ] }

  mapping = fastremap.inverse_component_map([ 1, 1, 1, 3 ], [ 4, 4, 5, 6 ]) 
  assert mapping == {1: [4, 5], 3: [6]}

  mapping = fastremap.inverse_component_map([ 1, 1, 1, 1 ], [ 4, 2, 5, 6 ]) 
  assert mapping == {1: [4, 2, 5, 6]}

  mapping = fastremap.inverse_component_map([], [])

def test_point_cloud():
  x = np.ones((2,2,2), dtype=np.uint8)
  ptc = fastremap.point_cloud(x)
  assert len(ptc) == 1

  gt = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
  ])
  assert np.all(ptc[1] == gt)

  x[0,0,0] = 2
  ptc = fastremap.point_cloud(x)
  assert len(ptc) == 2
  assert np.all(ptc[1] == gt[1:,:])
  assert np.all(ptc[2] == gt[:1,:])

  x[1,1,1] = 3
  ptc = fastremap.point_cloud(x)
  assert len(ptc) == 3
  assert np.all(ptc[1] == gt[1:7,:])
  assert np.all(ptc[2] == gt[:1,:])
  assert np.all(ptc[3] == gt[7:,:])  

  x = np.ones((0,0,0), dtype=np.uint8)
  ptc = fastremap.point_cloud(x)
  assert len(ptc) == 0

  

