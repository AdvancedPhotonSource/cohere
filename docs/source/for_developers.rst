=======
develop
=======
| This chapter has info for developers.

Installation for development
============================
Creating environment
++++++++++++++++++++
The best practice is to create coda environment allocated for the development. The Python version can be chosen by user:
::

    conda create -n cohere3.8 python=3.8
    conda activate cohere3.8


Pre-requisites
++++++++++++++
After activating conda environment install the packages/libraries listed below.
    - conda install tifffile -c conda-forge
    - if running AutoAlien1 algorithm during standard preprocessing: conda install scikit-learn -c conda-forge
    - if using Initial AI guess: conda install tensorflow -c conda-forge
    - if using cupy library: conda install cupy -c conda-forge
    - if using arrayfire: pip install arrayfire, install arrayfire library according to https://github.com/arrayfire/arrayfire-python/blob/master/README.md
Other packages have to be installed if running with cohere-ui package. Refer to :ref:`use` page, section "Installing Scripts".

Initialization
++++++++++++++
Clone the latest cohere repository from GitHub:
::

    git clone https://github.com/advancedPhotonSource/cohere
    cd cohere

| install:

::

    python setup.py develop

compile and install
+++++++++++++++++++
After changing code run the following command from 'cohere' directory:
::

    python setup.py develop

| Sometimes if the version number in setup.py script did not change, the install may not update. In this case remove build and distribute directories before running the setup.py script.

Adding new trigger
==================
The design allows to add a new feature in a standardized way. Typical feature is defined by a trigger and supporting parameters. The following modifications need to be done to add a new feature:
    - In cohere_core/controller/phasing.py, Rec.init, insert a new function to the to the iter_functions list in the correct order.
    - Implement the new function in cohere_core/controller/phasing.py, Rec class.
    - In cohere_core/controller/op_flow.py add code to add the new trigger row into flow array.
    - In cohere_core/controller/params.py add code to parse trigger and parameters applicable to the new feature.

Adding new algorithm
====================
todo

Conda Build
===========
For a new build change version in meta.yaml and setup.py files to the new version and run conda build:
::

    conda build -c conda-forge -c defaults .

- upload build to anaconda cloud

