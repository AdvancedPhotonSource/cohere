============
Installation
============
The cohere project can be installed on Linux, Windows and Mac operating systems.

Official Release Installation
=============================
It is recommended to install the cohere project packages in conda environment. Supported python versions are 3.10 - 3.11.

Create and activate conda environment::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py pyzmq
    conda activate <env_name>

The cohere project contains three computational modules: cohere_core, cohere_ui, cohere_beamlines, and a module containing examples data: cohere_examples.

The computational modules are installed as separate packages with pypi. The cohere_experiment package can be cloned or copied from the repository.
The instructions below show steps to install the cohere project.

PyPi installation::

    pip install cohere_core
    pip install cohere_ui
    pip install cohere_beamlines

The cohere_core package can be run utilizing numpy library. Other available libraries are cupy and torch.
User has to install the preferred library if using other than numpy.

If using cupy library::

    conda install cupy=12.2.0 -c conda-forge

If using torch library::

    pip install torch

Refer to :ref:`api_cohere_ui` page, for instructions how to use cohere with cohere_ui.



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

    pip install torch

After installation you may start using scripts as described in  :ref:`api_cohere_ui`, for example::

    cohere_gui

The examples are cloned as one of submodule to cohere project. The examples are in cohere_experiment subdirectory.

Beamlines specific installation
===============================
For Petra beamline install additional packages::

    pip install hdf5plugin pandas

Some of the modules can be installed with Pypi, as some can be cloned. A common scenario when developing on a beamline is installing release version of cohere_core and cohere_ui, and clone cohere_beamlines. Such arrangement offers reliability of the main modules and allows flexibility when dealing with changes on a beamline.

Examples installation
===============================
To obtain the examples from cohere_examples module, clone the repository and initialize it to align with your directory structure::

    git clone https://github.com/advancedPhotonSource/cohere_examples
    cd cohere_examples
    python init_examples.py

