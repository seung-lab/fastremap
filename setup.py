#!/usr/bin/env python
import setuptools
import numpy as np

# NOTE: If fastremap.cpp does not exist, you must run
# cython -3 --cplus fastremap.pyx

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  extras_require={
     ':python_version == "2.7"': ['futures'],
     ':python_version == "2.6"': ['futures'],
  },
  pbr=True,
  ext_modules=[
    setuptools.Extension(
      'fastremap',
      sources=[ 'fastremap.cpp' ],
      depends=[],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=[
        '-std=c++11', '-O3'
     ]
    ) 
  ],
  long_description_content_type='text/markdown',
)

