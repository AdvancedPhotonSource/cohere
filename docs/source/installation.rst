============
Installation
============
Currently we offer pip installation of the official release for Linux, Windows and Mac operating systems.

The latest development can be installed with cloning the "Dev" branch. Look at the "Latest development installation" section for instructions.

Conda Installation
==================
This will install latest official release.

First you must have `Conda <http://continuum.io/downloads>` installed.
It is highly recommended to install the cohere_core package in conda environment. Supported python versions are 3.6 - 3.11 for Linux and Mac, and 3.6 - 3.9 for Windows.
To create and activate the environment run the commands below::

    conda create --name <env_name> python=3.x -c conda-forge
    conda activate <env_name>
then run the installation command::

    pip install cohere-core
The cohere_core package can be run as installed running on numpy library. Other available libraries are cupy and torch.
User has to install the preferred library.

if using cupy library::

    conda install cupy -c conda-forge
Cohere-core package does not install python packages used by user's scripts in cohere-ui package. If planning to use the scripts Refer to :ref:`use` page, section "Installing Scripts".

Latest development installation
===============================
This will install the latest development. It might be not 100 percent tested. This installation might be right for a user that is looking for new feature that has not been released yet or want to try the latest development. We appreciate reporting any issues.

Create environment, activate it and clone cohere repository. It contains the cohere-ui submodule. Run the following commands::

    conda create --name <env_name> python=3.x -c conda-forge
    conda activate <env_name>
    git clone https://github.com/advancedPhotonSource/cohere --recurse-submodules
    cd cohere
    git checkout Dev
    pip install .
    cd cohere-ui
    git checkout Dev
    python setup.py
    sh cohere-ui/install_pkgs.sh    # for Linux and OS_X
    cohere-ui/install_pkgs.bat      # for Windows
For Windows make sure that numpy version is 1.23.5. The commands above will create conda environment and activate it, clone the packages, get the Dev branch, install, initialize. The install_pkgs script is interactive and the user must confirm the pacakages installations.
If using cupy library::

    conda install cupy -c conda-forge
After installation you may start using scripts from this directory, for example::

    python cohere-scripts/cdi_window.py
