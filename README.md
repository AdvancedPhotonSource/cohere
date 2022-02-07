[![Documentation Status](https://readthedocs.org/projects/cohere/badge/?version=latest)](http://cohere.readthedocs.io/en/latest/?badge=latest)

Project home page: https://cohere.readthedocs.io/

The cohere package provides tools for reconstruction of image of a nanoscale structures from data obtained using Bragg Coherent Diffraction Imaging technique.

The reconstruction has very good performance, in particular when utilizing GPU. User has a choice to run on cpu or GPU by selecting Python library. Supported libraries: numpy, cupy, arrayfire. The libraries cupy or arrayfire are installed by user. The solution offers concurrent processing for fast reconstruction of multiple starting points. 

Recently several features were added:
Genetic Algorithm (GA) - powerful feature that can deliver good reconstruction result by using GA principles.
Artificial Intelligence initial guess for reconstruction - uses AI to find reconstructed object that is subsequently used as input of further reconstruction.
AutoAlien1 algorithm - a method to remove aliens by automatic means during standard data preprocessing.

The tools offer a full solution from reading data, formatting the data, reconstruction, and visualization. The project was implemented for the Advanced Photon Source beamline 34-IDC and thus the data preparation and data visualization is customized for that specific beamline. The measurements and parameters that were used during the experiment are parsed specifically for the beamline setup. We offer the code, as the community would benefit from it when customizing for own case.

Author(s)

Barbara Frosik - Principal Software Engineer at Argonne National Laboratory

Ross Harder - Scientist at Argonne National Laboratory

License

Copyright (c) UChicago Argonne, LLC. All rights reserved. See LICENSE file.
