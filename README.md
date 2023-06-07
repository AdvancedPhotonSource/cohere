[![Documentation Status](https://readthedocs.org/projects/cohere/badge/?version=latest)](http://cohere.readthedocs.io/en/latest/?badge=latest)

Project home page: https://cohere.readthedocs.io/

The cohere package provides tools for reconstruction of image of a nanoscale structures from data obtained using Bragg Coherent Diffraction Imaging technique.

The reconstruction has very good performance, in particular when utilizing GPU. User has a choice to run on cpu or GPU by selecting Python library. Supported libraries: numpy, cupy, arrayfire. The libraries cupy or arrayfire are installed by user. The solution offers concurrent processing for fast reconstruction of multiple starting points. 

Important features:
- Genetic Algorithm (GA) - powerful feature that can deliver good reconstruction result by using GA principles. Based on research "Three-dimensional imaging of dislocation propagation during crystal growth and dissolution, Supplementary Information" by Jesse N. Clark et. al.
- Artificial Intelligence initial guess for reconstruction - uses AI to find reconstructed object that is subsequently used as input to further reconstruction. The work is built on the research by Yudong Yao, et. al: "AutoPhaseNN: Unsupervised Physics-aware Deep Learning of 3D Nanoscale Bragg Coherent Diffraction Imaging". 
A trained model must be provided when using this feature. User can download trained model by clicking the following link
https://g-29c18.fd635.8443.data.globus.org/cherukara/cohere-trained_model.hdf5
- AutoAlien1 algorithm - a method to remove aliens by automatic means during standard data preprocessing. Based on work "Removal of spurious data in Bragg coherent diffraction imaging: an algorithm for automated data preprocessing" by Kenley Pelzer et. al.
- Multipeak - support for an experiment where data is collected for adjacent peaks simultaneously and reconstructing this multipeak scenario. The research is in experimental stage. Implemented by Jason (Nick) Porter.

The tools offer a full solution for reading data, formatting the data, reconstruction, and visualization. Each of the componenet can be utilized independently. The project was implemented for the Advanced Photon Source beamline 34-ID-C and thus the data preparation and data visualization is customized for that specific beamline. The measurements and parameters that are used during the experiment are parsed specifically for the beamline setup. We offer the code, as the community would benefit from it when customizing for own case.

Author(s)

Barbara Frosik - Principal Software Engineer at Argonne National Laboratory

Ross Harder - Scientist at Argonne National Laboratory

License

Copyright (c) UChicago Argonne, LLC. All rights reserved. See LICENSE file.
