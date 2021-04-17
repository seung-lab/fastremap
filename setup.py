#!/usr/bin/env python
import setuptools
import sys

import numpy as np

# NOTE: If fastremap.cpp does not exist, you must run
# cython -3 --cplus fastremap.pyx

extra_compile_args = [
  '-std=c++11', '-O3', 
]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  python_requires="~=3.6", # >= 3.6 < 4.0
  pbr=True,
  ext_modules=[
    setuptools.Extension(
      'fastremap',
      sources=[ 'fastremap.cpp' ],
      depends=[],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=extra_compile_args,
    ) 
  ],
  long_description_content_type='text/markdown',
)

