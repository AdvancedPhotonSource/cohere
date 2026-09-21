[![Documentation Status](https://readthedocs.org/projects/cohere/badge/?version=latest)](http://cohere.readthedocs.io/en/latest/?badge=latest)

Project home page: [https://cohere.readthedocs.io/](https://cohere.readthedocs.io/)

The cohere package provides tools for reconstruction of image of a nanoscale structures from data obtained using Bragg Coherent Diffraction Imaging technique.

The reconstruction has very good performance, in particular when utilizing GPU. User has a choice to run on cpu or GPU by selecting Python library. Supported libraries: numpy, cupy, torch. The libraries cupy or torch are installed by user. The solution offers concurrent processing for fast reconstruction of multiple starting points. 

Important features:
- Genetic Algorithm (GA) - powerful feature that can deliver good reconstruction result by using GA principles. Based on research "Three-dimensional imaging of dislocation propagation during crystal growth and dissolution, Supplementary Information" by Jesse N. Clark et. al.
- Artificial Intelligence initial guess for reconstruction - uses AI to find reconstructed object that is subsequently used as input to further reconstruction. The work is built on the research by Yudong Yao, et. al: "AutoPhaseNN: Unsupervised Physics-aware Deep Learning of 3D Nanoscale Bragg Coherent Diffraction Imaging". 
A trained model must be provided when using this feature. User can download trained model by clicking the following link
https://g-29c18.fd635.8443.data.globus.org/cherukara/cohere-trained_model.hdf5
- AutoAlien1 algorithm - a method to remove aliens by automatic means during standard data preprocessing. Based on work "Removal of spurious data in Bragg coherent diffraction imaging: an algorithm for automated data preprocessing" by Kenley Pelzer et. al.
- Multipeak - support for an experiment where data is collected for adjacent peaks simultaneously and reconstructing this multipeak scenario. The research is in experimental stage. Implemented by Jason (Nick) Porter.
- chrono CDI - allows the oversampling requirement at each time step to be reduced. The increased time resolution will allow imaging of faster dynamics and of radiation-dose-sensitive samples. Based on work "Coherent diffractive imaging of time-evolving samples with improved temporal resolution" by A. Ulvestat et. al.

The tools offers a full solution for reading beamline specific experiment data, formatting the data, reconstruction, and visualization. Each of the components can be utilized independently. The project was implemented for the Advanced Photon Source beamline 34-ID-C.

Author(s)

Barbara Frosik - Principal Software Engineer at Argonne National Laboratory

Ross Harder - Scientist at Argonne National Laboratory

License

Copyright (c) UChicago Argonne, LLC. All rights reserved. See LICENSE file.
