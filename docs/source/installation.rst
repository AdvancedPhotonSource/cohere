============
Installation
============
Currently we are working on a new cohere release. In the mean time the project can be loaded in the development mode. 
The cohere project can be installed on Linux, Windows and Mac operating systems.

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

After installation you may start using scripts from this directory, for example::

    python cohere_ui/cohere_gui.py

The examples are cloned as one of submodule to cohere project. The examples are in cohere_experiment subdirectory.

Beamlines specific Installation
===============================
For Petra beamline install additional packages::

    pip install hdf5plugin pandas
