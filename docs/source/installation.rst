============
Installation
============
Currently we offer pip installation for Linux, Windows and Mac

Conda Installation
==================

First you must have `Conda <http://continuum.io/downloads>` installed.
It is highly recommended to install the cohere_core package in conda environment. Supported python versions are 3.6 - 3.10.
To create and activate the environment run the commands below::

    conda create --name <env_name> python=3.8
    conda activate <env_name>

then run the installation command::

    pip install cohere_core

The cohere_core package can be run as installed running on numpy library. Other available libraries are cupy and arrayfire.
Users has to install the library of theirs choice.

if using cupy library::

    conda install cupy -c conda-forge

if using arrayfire library::

    pip install arrayfire

and install arrayfire library according to https://github.com/arrayfire/arrayfire-python/blob/master/README.md

Cohere-core package does not install python packages used by user's scripts in cohere-ui package. If planning to use the scripts Refer to :ref:`use` page, section "Installing Scripts".
