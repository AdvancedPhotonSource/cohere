============
Installation
============
Currently we offer pip installation of the official release for Linux, Windows and Mac operating systems.

The latest development can be installed with cloning the "Dev" branch. Look at the "Latest development installation" section for instructions.

Conda Installation
==================
This will install latest official release.

It is highly recommended to install the cohere_core package in conda environment. Supported python versions are 3.10 - 3.11.

| Create and activate the environment.
Run the commands below::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py pyzmq
    pyzmq
    conda activate <env_name>
then run the cohere installation command::

    pip install cohere_core

The cohere_core package can be run utilizing numpy library. Other available libraries are cupy and torch.
User has to install the preferred library if other than numpy.

If using cupy library::

    conda install cupy=12.2.0 -c conda-forge
If torch is preferred, install with command::

    pip install torch
Refer to :ref:`use` page, for instructions how to use cohere with cohere_ui.

Note: The cupy installation on Windows may result in incompatible libraries, which makes the environment unusable. Run the repack.bat script from cohere-ui package and try running again.

.. _latest:
Latest development installation
===============================
This will install the latest development. It might be not 100 percent tested. This installation might be right for a user that is looking for new feature that has not been released yet or want to try the latest development. We appreciate reporting any issues.

Create environment, activate it and clone cohere repository. It contains the cohere-ui submodule. Run the following commands::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py
    conda activate <env_name>
    git clone https://github.com/advancedPhotonSource/cohere --recurse-submodules
    cd cohere
    git checkout Dev
    pip install -e .
    cd cohere-ui
    git checkout Dev
    pip install -e .
If using cupy library::

    conda install cupy=12.2.0 -c conda-forge
If using torch library::

    pip install torch
For Petra beamline install additional package::

    pip install hdf5plugin pandas
After installation you may start using scripts from this directory, for example::

    python cohere_ui/cohere_gui.py
