project(
  'fastremap',
  'cpp',
  version: '1.20.1',
  default_options: ['cpp_std=c++11', 'optimization=3'],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)

cython = find_program('cython')

# Generate fastremap.cpp from fastremap.pyx using cython -3 --cplus
fastremap_cpp = custom_target(
  'fastremap_cpp',
  output: 'fastremap.cpp',
  input: 'fastremap/fastremap.pyx',
  command: [cython, '-3', '--cplus', '-o', '@OUTPUT@', '@INPUT@'],
)

incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

extra_args = [
  '-DNDEBUG=1',
  '-DNPY_NO_DEPRECATED_API=1',
  '-DNPY_1_7_API_VERSION=1',
]

if host_machine.system() == 'darwin'
  extra_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
endif

py.extension_module(
  'fastremap',
  fastremap_cpp,
  subdir: 'fastremap',
  include_directories: include_directories('fastremap', incdir_numpy),
  cpp_args: extra_args,
  install: true,
)

py.install_sources(
  'fastremap/__init__.py',
  'fastremap/fastremap.pxd',
  'fastremap/fastremap.pyi',
  'fastremap/py.typed',
  subdir: 'fastremap',
)
