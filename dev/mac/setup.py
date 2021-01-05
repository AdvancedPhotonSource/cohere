import setuptools
from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules=cythonize(
    ["cohere/src_py/cyth/bridge_cpu.pyx", "cohere/src_py/cyth/bridge_opencl.pyx", ],),
      name='cohere',
      author = 'Barbara Frosik, Ross Harder',
      author_email = 'bfrosik@anl.gov',
      url='https://github.com/advancedPhotonSource/cdi',
      version='1.4',
      packages=setuptools.find_packages(),
      package_data={'cohere' : ['*.pyx','*.so'], 'cohere.src_py.cyth' : ['*.pyx','*.so']}
)
