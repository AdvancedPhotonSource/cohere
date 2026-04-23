============
Installation
============
The cohere project can be installed on Linux, Windows and Mac operating systems.

Official Release Installation
=============================
Create and activate conda environment::

    conda create -n <env_name> -c conda-forge python=3.14
    conda activate <env_name>

Note: When installing the cohere project with newer python versions it is possible that some dependencies may not be available. In this case, the user can install the necessary packages using conda before installing the cohere project. We found the following packages needed to be installed.::

    mamba install -c conda-forge mpi4py pyzmq  # Linux
    conda install -c conda-forge PyQt6  # Linux
    conda install -c conda-forge mpi4py pyzmq xrayutilities  # Mac
    conda install -c conda-forge mpi4py pyzmq xrayutilities matplotlib  # Windows

The cohere project consists of three modules and each of them is a separate package in PyPi.::

    pip install cohere_core cohere_ui cohere_beamlines

If using cupy library (for Linux and Windows)::

    pip install cupy-cuda13x    # where cuda major version is 13

If using torch library (Linux and Windows; torch installation is included in cohere_core Pypi installation for Mac).
Install the nightly version with the corresponding CUDA driver (example CUDA 13.2) support using the following command::

    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
    Note: make sure your CUDA driver matches this command

After installation you may start using scripts as described in  :ref:`api_cohere_ui`, for example::

    cohere_gui
    beamline_preprocess <dir_to_cohere_examples>/example_workspace/scan_54
    standard_preprocess <dir_to_cohere_examples>/example_workspace/scan_54
    cohere_reconstruction <dir_to_cohere_examples>/example_workspace/scan_54
    beamline_postprocess <dir_to_cohere_examples>/example_workspace/scan_54

The cohere project provides examples in the cohere_examples submodule. The examples contain experiment data and configuration files in defined directory structure. To get the examples refer to :ref:`examples`

.. _latest:

Latest development installation
===============================
This will install the latest development. This installation might be right for a user that is looking for the latest development or do some development.

Create conda environment, activate it and clone cohere repository. Cohere project has cohere_core module and three submodules: cohere-ui, cohere_beamlines, and cohere_examples.
Run the following commands::

    conda create -n <env_name> -c conda-forge python=3.14
    conda activate <env_name>

    Note: When installing the cohere project with newer python versions it is possible that some dependencies may not be available. In this case, the user can install the necessary packages using conda before installing the cohere project. We found the following packages needed to be installed.

    mamba install -c conda-forge mpi4py pyzmq  # Linux
    conda install -c conda-forge PyQt6  # Linux
    conda install -c conda-forge mpi4py pyzmq xrayutilities  # Mac
    conda install -c conda-forge mpi4py pyzmq xrayutilities matplotlib  # Windows

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

If using cupy library (for Linux and Windows)::

    pip install cupy-cuda13x    # where cuda major version is 13

If using torch library (Linux and Windows; torch installation is included in cohere_core Pypi installation for Mac).
Install the nightly version with the corresponding CUDA driver (example CUDA 13.2) support using the following command::

    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
    Note: make sure your CUDA driver matches this command

After installation you may start using scripts as described in  :ref:`api_cohere_ui`, for example::

    cohere_gui

Cohere examples are cloned as one of submodule to cohere project. The examples are in cohere_experiment subdirectory.

Beamlines specific installation
===============================
For Petra beamline install additional packages::

    pip install hdf5plugin

Some of the modules can be installed with Pypi, as some can be cloned. A common scenario when developing on a beamline is installing release version of cohere_core and cohere_ui, and clone cohere_beamlines. Such arrangement offers reliability of the main modules and allows flexibility when dealing with changes on a beamline.

.. _examples:

Cohere examples installation
============================
To obtain the examples from cohere_examples module, clone the repository and initialize it to align with your directory structure::

    git clone https://github.com/advancedPhotonSource/cohere_examples
    cd cohere_examples
    python init_examples.py
