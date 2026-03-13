.. _api_cohere_ui:

=========
cohere_ui
=========

| Contents:

cohere_gui
----------
This script provides GUI interface to cohere tools. User can create new cohere experiment or load an existing one. The GUI offers user friendly interface that allows to define configuration, set parameters to defaults, and run the scripts: beamline_preprocessing, standard_preprocessing, run_rec, and beamline_visualization, with push buttons.

To run this script from command line::

    cohere_gui

optional arguments may follow:  --no_verify, --debug

Use --help to get explanation of command line parameters.

beamline_preprocess
-------------------
.. automodule:: cohere_ui.beamline_preprocess

standard_preprocess
-------------------
.. automodule:: cohere_ui.standard_preprocess

cohere_reconstruction
---------------------
.. automodule:: cohere_ui.cohere_reconstruction

beamline_postprocess
--------------------
.. automodule:: cohere_ui.beamline_postprocess

cohere_full
-----------
.. automodule:: cohere_ui.cohere_full

simple_phasing
--------------
.. automodule:: cohere_ui.simple_phasing

create_aps34idc_experiment
--------------------------
.. automodule:: cohere_ui.create_aps34idc_experiment

copy_setup
----------
.. automodule:: cohere_ui.copy_setup