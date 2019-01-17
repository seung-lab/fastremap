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






