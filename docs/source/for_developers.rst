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
   - conda install Cython
   - pip install tifffile
   - pip install pylibconfig2
   - pip install GPUtil
   - pip install pyparsing
   - pip install mayavi
   - pip install xrayutilities
   - pip install psutil
   - conda install pyqt
   - pip install sklearn

| Install ArrayFire library version 3.6.2 (available at http://arrayfire.s3.amazonaws.com/index.html#!/3.6.2%2F). 
| During installation respond y/y, so the 'arrayfire' directory will be created in <cwd>, where <cwd> is the current directory. The arrayfire installation directory will be <cwd>/arrayfire.

Initialization
++++++++++++++
| Clone the latest cohere repository from GitHub:
::

    git clone https://github.com/advancedPhotonSource/cohere
    cd cohere

| Copy development scripts
::

    cp dev/linux/* .     // for linux
    cp dev/mac/* .       //for mac


| Initialize with interactive script. 
::

    source init.sh

| The script prompts for path to ArrayFire installation directory. 
| The script does the following:
- sets the compilation library and include directory to the entered directories
- compiles 
- installs in conda environment
- sets the LD_LIBRARY_PATH/DYLD_LIBRARY_PATH in this session
- sets the LD_LIBRARY_PATH/DYLD_LIBRARY_PATH in the setenv.sh script 

| If the libraries are at different location than the installation (for example moved to lib directory), then the init script will not work. In this case edit setup.py file to correct lib directory and include directory. 
| Compile and install:

::

    python setup.py install

| and set the LD_LIBRARY_PATH/DYLD_LIBRARY_PATH in this session and in setenv.sh script manually

compile and install
+++++++++++++++++++
| After changing code run the following command from 'cohere' directory:
::

    python setup.py install

| Sometimes if the version number in setup.py script did not change, the install may not update. in this case remove build and distribute directories before running the setup.py script.

Activating environment
===================
| When running in development environment the libraries are not loaded into conda location. Therefore the LD_LIBRARY_PATH/DYLD_LIBRARY_PATH must include path to arrayfire libraries.
| Run the following command to set the environment variable when opening a new terminal:
::

    source setenv.sh

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

- Copy development scripts/files
::

    cp dev/linux/* .     // for linux
    cp dev/mac/* .       //for mac

- change version in meta.yaml and setup.py files to the new version


- run conda build:
::

    conda build -c conda-forge -c bfrosik -c defaults .

- upload build to anaconda cloud

