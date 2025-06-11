============
Installation
============
Currently we offer pip installation of the official release for Linux, Windows and Mac operating systems.

The latest development can be installed with cloning the "Dev" branch. Look at the "Latest development installation" section for instructions.

Conda Installation
==================
This will install latest official release.

First you must have `Conda <http://continuum.io/downloads>` installed.
It is highly recommended to install the cohere_core package in conda environment. Supported python versions are 3.6 - 3.10.

| Create and activate the environment.
Run the commands below::

    conda create -n <env_name> -c conda-forge python=3.11 mpi4py=3.1.5
    conda activate <env_name>
then run the cohere installation command::

    pip install cohere-core
| Additional packages must be installed when using cohere-iu, a complementing user interface.
Run the commands below to install the additional packages::

    pip install pyqt5 mayavi scikit-image xrayutilities vtk==9.3.1 scipy==1.14.1
The cohere_core package can be run utilizing numpy library. Other available libraries are cupy and torch.
User has to install the preferred library.

If using cupy library::

    conda install cupy=12.2.0 -c conda-forge
If torch is preferred, install with command::

    pip install torch
Refer to :ref:`use` page, for instructions how to use cohere with cohere-ui.

Note: The cupy installation on Windows may result in incompatible libraries, which makes the environment unusable. Run the repack.bat script from cohere-ui package and try running again.

Latest development installation
===============================
This will install the latest development. It might be not 100 percent tested. This installation might be right for a user that is looking for new feature that has not been released yet or want to try the latest development. We appreciate reporting any issues.

Create environment, activate it and clone cohere repository. It contains the cohere-ui submodule. Run the following commands::

    conda create --name <env_name> -c conda-forge python=3.11 mpi4py=3.1.5
    conda activate <env_name>
    git clone https://github.com/advancedPhotonSource/cohere --recurse-submodules
    cd cohere
    git checkout Dev
    pip install .
    pip install pyqt5 mayavi scikit-image xrayutilities pyvista vtk==9.3.1 scipy==1.14.1
    cd cohere-ui
    git checkout Dev
    python setup.py
For Windows make sure that numpy version is 1.23.5. The commands above will create conda environment and activate it, clone the packages, get the Dev branch, install, initialize.
| If using cupy library::

    conda install cupy=12.2.0 -c conda-forge
| If using torch library::

    pip install torch
| for Petra beamline install additional package::

    pip install hdf5plugin
After installation you may start using scripts from this directory, for example::

    python cohere_ui/cdi_window.py
