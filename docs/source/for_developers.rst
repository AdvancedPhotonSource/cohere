=======
develop
=======
| This chapter has info for developers.

Adding a new trigger
====================
The design allows to add a new feature in a structured way. Typical feature is defined by a trigger and supporting parameters. Refer to  :ref:`formula` for the design description. The following modifications/additions are needed when adding a new feature named <new_feature>:
    - In cohere_core/controller/phasing.py, Rec constructor, insert a new function name '<new_feature>_operation' to the self.iter_functions list in the correct order. Note: If the order of functions executed during iterations should be different, one can subclass the Rec class and write own constructor overriding the 'iter_functions' field.
    - Implement the '<new_feature>_operation' function in cohere_core/controller/phasing.py, Rec class. Add necessary parameters.
    - In cohere_core/controller/phasing.py add code to set any new defaults if applicable when creating Rec object.
    - In utilities/config_verifier.py add code to verify added parameters.
    - The feature will be available in cohere_core and can be used from command line interface. If the GUI support is wanted, modify the cohere_gui.py file to add the feature.

Adding a new sub-trigger
========================
If the new feature will be used in a context of sub-triggers, in addition to the above steps, the following modifications/additions need to be done:
    - Assign a mnemonics for the feature that will be used to identify it in the algorithm sequence.
    - In cohere_core/controller/op_flow.py add entry in the 'sub_triggers' dictionary, where key is the arbitrary assigned mnemonics, and value is the trigger name.
    - In cohere_core/controller/features.py add new feature with the following steps:

       - Add feature class that inherits from 'TriggeredOp' class. One feature can have different method, including different parameters. Therefore, the feature class may contain multiple classes with different implementations, as aggregate, or at least one class.
       - The aggregate class(es) should implement the 'apply_trigger' function with the trigger code.
       - The feature class should have implemented 'create_obj' function that creates the aggregate object(s) according to parameters. It is recommended that parameters are verified and exception raised in case of error.
       - Add code in the constructor factory function 'create' to construct the new feature object.

    - In cohere_core/controller/phasing.py, 'init_iter_loop' function, add the new feature object, under comment: ' # create the trgger/sub-trigger objects' created using constructor factory, the same way as shrink_wrap_obj, and other features.
    - In cohere_core/controller/phasing.py, Rec class add the trigger function. The code inside should call the trigger on the feature object with args.

       | Note: The easiest way to implement the feature is to copy one already implemented and modify following the recipe above.

Adding a new algorithm
======================
The algorithm sequence defines functions executed during modulus projection and during modulus stages. Adding new algorithm requires the following steps:
    - In cohere_core/controller/op_flow.py add entry in the 'algs' dictionary, where key is the mnemonic used in algorithm_sequence, and value is the tuple defining functions, ex: 'ER': ('to_reciprocal_space', 'modulus', 'to_direct_space', 'er')
    - In cohere_core/controller/phasing.py, Rec constructor, insert a new function name, implementation of the new algorithm, to the self.iter_functions list in the correct order.
    - In cohere_core/controller/phasing.py, Rec class add the new algorithm function(s).

Pypi Build
==========
For a new build change version in and pyproject.toml files to the new version and run pypi build:

  ::

    python3 -m pip install twine
    python3 -m build --sdist
    python3 -m build --wheel
    twine check dist/*

| Upload to the test server and test

  ::

    twine upload --repository testpypi dist/*

| Test in a new environment

  ::

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cohere_core --user

| Test Linux, Windows, and Mac

| upload build to pypi cloud

  ::

    twine upload dist/*

