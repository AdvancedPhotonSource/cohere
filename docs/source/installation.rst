============
Installation
============
The cohere project can be installed on Linux, Windows and Mac operating systems.

Official Release Installation
=============================
Create and activate conda environment::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py pyzmq
    conda activate <env_name>

The cohere project consists of three modules and each of them is a separate package in PyPi.::

    pip install cohere_core cohere_ui cohere_beamlines

Note: If the user's choice is own beamline dependent preprocessing and postprocessing and cohere_core package as a reconstruction engine, the user can access cohere_core APIs or use cohere_ui user's scripts. In the first case only the cohere_core package has to be installed, and in the second case, the cohere_ui has to be added.

If using cupy library::

    conda install cupy=12.2.0 -c conda-forge

If using torch library::

    pip install torch torchvision torchaudio

After installation you may start using scripts as described in  :ref:`api_cohere_ui`, for example::

    cohere_gui
    beamline_preprocess <dir_to_cohere_examples>/example_workspace/scan_54
    standard_preprocess <dir_to_cohere_examples>/example_workspace/scan_54
    run_reconstruction <dir_to_cohere_examples>/example_workspace/scan_54
    beamline_visualization <dir_to_cohere_examples>/example_workspace/scan_54

The cohere project provides examples in the cohere_examples submodule. The examples contain experiment data and configuration files in defined directory structure. To get the examples refer to :ref:`examples`

.. _latest:

Latest development installation
===============================
This will install the latest development. This installation might be right for a user that is looking for the latest development or do some development.

Create conda environment, activate it and clone cohere repository. Cohere project has cohere_core module and three submodules: cohere-ui, cohere_beamlines, and cohere_examples.
Run the following commands::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py pyzmq
    conda activate <env_name>
    git clone https://github.com/advancedPhotonSource/cohere --recurse-submodules
    cd cohere
    git checkout Dev
    pip install -e .   # include the -e option if you intend to edit cohere_core
    cd cohere_beamlines
    git checkout Dev
    pip install -e .   # include the -e option if you intend to edit cohere_beamlines
    cd ../cohere-ui
    git checkout Dev
    pip install -e .   # include the -e option if you intend to edit cohere_ui

If using cupy library::

    conda install cupy=12.2.0 -c conda-forge

If using torch library::

    pip install torch torchvision torchaudio

After installation you may start using scripts as described in  :ref:`api_cohere_ui`, for example::

    cohere_gui

The examples are cloned as one of submodule to cohere project. The examples are in cohere_experiment subdirectory.

Beamlines specific installation
===============================
For Petra beamline install additional packages::

    pip install hdf5plugin pandas

Some of the modules can be installed with Pypi, as some can be cloned. A common scenario when developing on a beamline is installing release version of cohere_core and cohere_ui, and clone cohere_beamlines. Such arrangement offers reliability of the main modules and allows flexibility when dealing with changes on a beamline.

.. _examples:

Examples installation
=====================
To obtain the examples from cohere_examples module, clone the repository and initialize it to align with your directory structure::

    git clone https://github.com/advancedPhotonSource/cohere_examples
    cd cohere_examples
    python init_examples.py

