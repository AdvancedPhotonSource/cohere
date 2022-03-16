============
Installation
============
Currently we offer conda installation for Linux and Mac

Conda Installation
==================

First you must have `Conda <http://continuum.io/downloads>` installed.
It is highly recommended to install the cohere package in conda environment.
To create and activate the environment run the commands below::
    conda create --name <env_name> python=3.9
    conda activate <env_name>

then run the installation command::

    conda install cohere -c bfrosik -c conda-forge

The cohere package can be run as installed running on numpy library. Other available libraries are cupy and arrayfire.
User has to intall the library of his choice.

if using cupy library::

    conda install cohere -c bfrosik -c conda-forge

if using arrayfire library::

    pip install arrayfire

install arrayfire library according to https://github.com/arrayfire/arrayfire-python/blob/master/README.md
