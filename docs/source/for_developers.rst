=======
develop
=======
| This chapter has info for developers

Installation for development
============================
Creating environment
++++++++++++++++++++
| The best practice is to create coda environment allocated for the development:
::

    conda create -n cohere3.7.6 python=3.7.6
    conda activate cohere


Pre-requisites
++++++++++++++
| After activating conda environment install the packages/libraries listed below.
- Python packages installation:
   - pip install tifffile
   - pip install pylibconfig2
   - pip install GPUtil
   - pip install pyparsing
   - conda install mayavi -c conda-forge
   - pip install xrayutilities
   - pip install psutil
   - conda install pyqt
   - if running AutoAlien1 algorithm during standard preprocessing: pip install sklearn
   - if using cupy library: conda install cupy
   - if using arrayfire: install according to https://github.com/arrayfire/arrayfire-python/blob/master/README.md

Initialization
++++++++++++++
| Clone the latest cohere repository from GitHub:
::

    git clone https://github.com/advancedPhotonSource/cohere
    cd cohere

| install:

::

    python setup.py install

compile and install
+++++++++++++++++++
| After changing code run the following command from 'cohere' directory:
::

    python setup.py install

| Sometimes if the version number in setup.py script did not change, the install may not update. in this case remove build and distribute directories before running the setup.py script.

Adding new trigger
==================
| The design applied in c++ code allows to add a new feature in a standardized way. Each feature is defined by a trigger and supporting parameters. The following modifications need to be done to add a new feature:
- In cohere/include/common.h file insert a new definition for the flow_item to the flow_def array in the correct order.
- Update the flow_seq_len defined in cohere/include/common.h (i.e. increase by 1).
- Add code to parse feature's parameters in cohere/include/parameters.hpp and cohere/src_cpp/parameters.cpp.
- Add the new function to the cohere/include/worker.hpp and cohere/src_cpp/worker.cpp
- add the pair (func_name, fp) to the flow_ptr_map in worker.cpp.

Adding new algorithm
====================
| Currently two algorithms are supported: ER and HIO.

Conda Build
===========
- In the cohere directory create "lib" and "include" directories. Copy content of <arrayfire installation directory>/lib64 to lib directory. Copy content of <arrayfire installation directory>/include to include directory. 

- change version in meta.yaml and setup.py files to the new version


- run conda build:
::

    conda build -c conda-forge -c bfrosik -c defaults .

- upload build to anaconda cloud

