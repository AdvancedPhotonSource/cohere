=======
develop
=======
| This chapter has info for developers

Installation for development
============================
Creating environment
++++++++++++++++++++
| The best practice is to create coda environment allocated for the development. The Python version can be chosen by user:
::

    conda create -n cohere3.9 python=3.9
    conda activate cohere3.9


Pre-requisites
++++++++++++++
| After activating conda environment install the packages/libraries listed below.
- Python packages installation:
   - conda install tifffile -c conda-forge
   - conda install gputil -c conda-forge
   - conda install mayavi -c conda-forge
   - conda install xrayutilities -c conda-forge
   - conda install psutil -c conda-forge
   - if running AutoAlien1 algorithm during standard preprocessing: pip install sklearn
   - if using Initial AI guess: conda install tensorflow -c conda-forge
   - if using cupy library: conda install cupy -c conda-forge
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

| Sometimes if the version number in setup.py script did not change, the install may not update. In this case remove build and distribute directories before running the setup.py script.

Adding new trigger
==================
| The design allows to add a new feature in a standardized way. Typical feature is defined by a trigger and supporting parameters. The following modifications need to be done to add a new feature:
- In cohere/controller/phasing.py, Rec.init, insert a new function to the to the iter_functions list in the correct order.
- Implement the new function in cohere/controller/phasing.py, Rec class.
- In cohere/controller/op_flow.py add code to add the new trigger row into flow array.
- In cohere/controller/params.py add code to parse trigger and parameters applicable to the new feature.

Adding new algorithm
====================
| todo

Conda Build
===========
- change version in meta.yaml and setup.py files to the new version


- run conda build:
::

    conda build -c conda-forge -c defaults .

- upload build to anaconda cloud

