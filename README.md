[![Documentation Status](https://readthedocs.org/projects/cohere/badge/?version=latest)](http://cohere.readthedocs.io/en/latest/?badge=latest)

Project home page: https://cohere-legacy.readthedocs.io/

The cohere version 1.x.x is a legacy project. It is functional and available but there will be no new development in this branch. User can access the legacy documentation (link above) to find how to install and use this product.
A continuation with new development is carried on in main branch with versions starting at 2.0.

The cohere package provides tools for reconstruction of image of a nanoscale structures from data obtained using Bragg Coherent Diffraction Imaging technique.

The reconstruction has very good performance, in particular when utilizing GPU. User has a choice to run on cpu or GPU (cuda or OpenCL library). The solution offers concurrent processing for fast reconstruction of multiple starting points. 

A powerful feature that can deliver good reconstruction result offered by the cohere package is genetic algorithm (GA).

The tools offer a full solution from reading data, formatting the data, reconstruction, and visualization. The project was implemented for the Advanced Photon Source beamline 34-IDC and thus the data preparation and data visualization is customized for that specific beamline. The measurements and parameters that were used during the experiment are parsed specifically for the beamline setup. We offer the code, as the community would benefit from it when customizing for own case.

Author(s)

Barbara Frosik - Principal Software Engineer at Argonne National Laboratory

Ross Harder - Scientist at Argonne National Laboratory

License

Copyright (c) UChicago Argonne, LLC. All rights reserved. See LICENSE file.

C++ Libraries

    ArrayFire open source
