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

  data = np.arange(8).astype(np.bool)
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

def test_3d_renumber():
  for dtype in DTYPES:
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

  big = np.random.randint(0, (2**64)-1, size=(512,512,100), dtype=np.uint64)
  big, remapdict = fastremap.renumber(big, preserve_zero=True, in_place=True)
  assert np.dtype(big.dtype).itemsize <= 4
  assert np.dtype(big.dtype).itemsize > 1


def test_remap_1d():
  for dtype in DTYPES:
    print(dtype)
    data = np.array([1, 2, 3, 4, 5], dtype=dtype)
    remap = {
      1: 10,
      2: 30,
      3: 15,
      4: 0,
      5: 5,
    }

    result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=False)
    assert np.all(result == [10, 30, 15, 0, 5])

    del remap[2]
    try:
      result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=False)
      assert False
    except KeyError:
      pass 

    result = fastremap.remap(np.copy(data), remap, preserve_missing_labels=True)
    assert np.all(result == [10, 2, 15, 0, 5])

def test_remap_2d():
  for dtype in DTYPES:
    print(dtype)
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

def test_mask():
  for dtype in DTYPES:
    for in_place in (True, False):
      print(dtype)
      data = np.arange(100, dtype=dtype)
      data = fastremap.mask(data, [5, 10, 15, 20], in_place=in_place)

      labels, cts = np.unique(data, return_counts=True)
      assert cts[0] == 5 
      assert labels[0] == 0
      assert np.all(cts[1:] == 1)
      assert len(labels == 95)

def test_mask_except():
  for dtype in DTYPES:
    for in_place in (True, False):
      for value in (0, 7):
        print(dtype)
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

def test_asfortranarray():
  dtypes = list(DTYPES) + [ np.float32, np.float64, np.bool ]
  for dtype in dtypes:
    print(dtype)
    for dim in (1, 4, 7, 9, 27, 31, 100, 127, 200):
      print(dim)
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


def test_ascontiguousarray():
  dtypes = list(DTYPES) + [ np.float32, np.float64, np.bool ]
  for dtype in dtypes:
    for dim in (1, 4, 7, 9, 27, 31, 100, 127, 200):
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