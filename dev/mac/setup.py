from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize

c_sources = ['cohere/src_cpp/bridge.cpp', 'cohere/src_cpp/manager.cpp', 'cohere/src_cpp/parameters.cpp', 'cohere/src_cpp/pcdi.cpp', 'cohere/src_cpp/resolution.cpp', 'cohere/src_cpp/state.cpp', 'cohere/src_cpp/support.cpp', 'cohere/src_cpp/util.cpp', 'cohere/src_cpp/worker.cpp']

exts = [
    Extension('cohere.src_py.cyth.bridge_opencl',
    sources = ['cohere/src_py/cyth/bridge_opencl.pyx'] + c_sources,
    extra_compile_args = ["-std=c++11"],
    language='c++11',
    libraries = ['afopencl', ],
    library_dirs = ['lib',],
    include_dirs = ['cohere/include', 'include',],
    language_level = 3
    ),

    Extension('cohere.src_py.cyth.bridge_cpu',
    sources = ['cohere/src_py/cyth/bridge_cpu.pyx'] + c_sources,
    extra_compile_args = ["-std=c++11"],
    language='c++11',
    libraries = ['afcpu', ],
    library_dirs = ['lib',],
    include_dirs = ['cohere/include', 'include',],
    language_level = 3
    ),
    ]


setup(
      ext_modules = cythonize(exts),
      name = 'cohere',
      author = 'Barbara Frosik, Ross Harder',
      author_email = 'bfrosik@anl.gov',
      url='https://github.com/advancedPhotonSource/cohere',
      version = '1.2',
      packages = find_packages(),
      package_data = {'cohere' : ['*.pyx','*.so'], 'cohere.src_py.cyth' : ['*.pyx','*.so']}
)
