============
Installation
============
The cohere project can be installed on Linux, Windows and Mac operating systems.

Official Release Installation
=============================
It is recommended to install the cohere_core package in conda environment. Supported python versions are 3.10 - 3.11.

Create and activate conda environment::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py pyzmq
    conda activate <env_name>

Install cohere_core::

    pip install cohere_core

The cohere_ui package provides user interface that supports several beamlines and offers user friendly solution to interact with cohere_core::

    pip install cohere_ui

The cohere_core package can be run utilizing numpy library. Other available libraries are cupy and torch.
User has to install the preferred library if other than numpy.

If using cupy library::

    conda install cupy=12.2.0 -c conda-forge

If using torch library::

    pip install torch

Refer to :ref:`api_cohere_ui` page, for instructions how to use cohere with cohere_ui.

Note: The cupy installation on Windows may result in incompatible libraries, which makes the environment unusable. Run the repack.bat script from cohere-ui package and try running again.

Cohere project provides examples for several beamlines in cohere_examples repository. To obtain the examples, clone the repository and initialize it to align with your directory structure::

    git clone https://github.com/advancedPhotonSource/cohere_examples
    cd cohere_examples
    python init_examples.py

.. _latest:

Latest development installation
===============================
This will install the latest development. This installation might be right for a user that is looking for the latest development or do some development.

Create conda environment, activate it and clone cohere repository. Cohere project has cohere_core module and two submodules: cohere-ui and cohere_examples.
Run the following commands::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py pyzmq
    conda activate <env_name>
    git clone https://github.com/advancedPhotonSource/cohere --recurse-submodules
    cd cohere
    git checkout Dev
    pip install -e .   # include the -e option if you intend to edit cohere_core
    cd cohere-ui
    git checkout Dev
    pip install -e .   # include the -e option if you intend to edit cohere_ui

If using cupy library::

    conda install cupy=12.2.0 -c conda-forge

If using torch library::

    pip install torch

After installation you may start using scripts from this directory, for example::

    python cohere_ui/cohere_gui.py

The examples are cloned as one of submodule to cohere project. The examples are in cohere_experiment subdirectory.

Beamlines specific Installation
===============================
For Petra beamline install additional packages::

    pip install hdf5plugin pandas
