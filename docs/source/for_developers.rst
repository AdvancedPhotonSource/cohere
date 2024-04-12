=======
develop
=======
| This chapter has info for developers.

Installation for development
============================
The best practice is to create conda environment allocated for the development.

  ::

    conda create -y -n <dev_env> -c conda-forge python=3.10 mpi4py=3.1.5 gputil psutil mayavi pyqt scikit-image xrayutilities
    conda activate <dev_env>

| Clone the latest cohere repository from GitHub. This will include the cohere-ui directory with all of the cohere-ui content, such users scripts and example.

  ::

    git clone https://github.com/advancedPhotonSource/cohere --recurse-submodules

| After the package is installed checkout the Dev environment and create your branch. Then install the cohere into the environment:

  ::

    cd cohere
    git checkout Dev
    git checkout -b <dev_branch>
    pip install -e .

| Go to cohere-ui directory and checkout the Dev branch and create your own branch, then run setup.py. The setup.py script modifies paths from relative to absolute in the provided example configuration.

  ::

    cd cohere-ui
    git checkout Dev
    git checkout -b <dev_branch>
    python setup.py

| If planning to use GPUs, install the packages/libraries that you wish to use.

  ::

    conda install -y cupy=12.2.0 -c conda-forge # if using cupy library
    pip install torch # if using torch

Adding new trigger
==================
The design allows to add a new feature in a standardized way. Typical feature is defined by a trigger and supporting parameters. The following modifications/additions need to be done to add a new feature:
    - In cohere_core/controller/phasing.py, Rec constructor, insert a new function name ending with '_operation' to the self.iter_functions list in the correct order.
    - Implement the new trigger function in cohere_core/controller/phasing.py, Rec class.
    - In cohere_core/controller/phasing.py add code to set any new defaults when creating Rec object.
    - In utilities/config_verifier.py add code to verify added parameters.

Adding new feature for sub-trigger
==================================
If the new feature will be used in a context of sub-triggers, in addition to the above steps, the following modifications/additions need to be done:
    - In cohere_core/controller/op_flow.py add entry in the sub_triggers dictionary, where key is the arbitrary assigned mnemonics, and value is the trigger name.
    - In cohere_core/controller/phasing.py, Rec.init function, create_feat_objects sub-function, add the new feature object, created the same as shrink_wrap_obj, and other features.
    - In cohere_core/controller/phasing.py, Rec class add the trigger function. The code inside should call the trigger on the feature object with *args.
    - in cohere_core/controller/features.py add new feature class.

       | The constructor factory function create should have a new lines to construct the new object.
       | The feature class should be subclass of Feature and
       | should have implemented create_obj function that creates sub-object(s) and
       | should have defined the sub-object(s) class(es). The embedded class contains the apply_trigger function that has the trigger code. Some features can be configured to different types and therefore multiple classes can be defined.
       |
       | The easiest way to implement the feature is to copy one already implemented and modify.

Adding new algorithm
====================
The algorithm sequence defines functions executed during modulus projection and during modulus. Adding new algorithm requires the following steps:
    - In cohere_core/controller/op_flow.py add entry in the algs dictionary, where key is the mnemonic used in algorithm_sequence, and value is the tuple defining functions, ex: 'ER': ('to_reciprocal_space', 'modulus', 'to_direct_space', 'er')
    - In cohere_core/controller/phasing.py, Rec constructor, insert a new function name to the self.iter_functions list in the correct order.
    - In cohere_core/controller/phasing.py, Rec class add the new algorithm function(s).

Pypi Build
==========
For a new build change version in and setup.py files to the new version and run pypi build:

  ::

    pip install .
    python setup.py check
    python setup.py sdist
    python setup.py bdist_wheel --universal

| Upload to the test server and test

  ::

    pip install twine
    twine upload --repository testpypi dist/*

| Test in a new environment

  ::

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cohere_core --user

| Test Linux, Windows, and Mac

| upload build to pypi cloud

  ::

    twine upload dist/*

