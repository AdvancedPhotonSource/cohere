=====
About
=====

The cohere package provides tools for reconstruction of image of a nanoscale structures from data obtained using Bragg Coherent Diffraction Imaging technique.

The reconstruction has very good performance, in particular when utilizing GPU. User has a choice to run on cpu or GPU by choosing the processing library. The solution offers concurrent processing for fast reconstruction of multiple starting points.

A powerful features that can deliver good reconstruction result offered by the cohere package is genetic algorithm (GA) or AI guess.

The tools are supplemented by another package cohere-ui which contains users scripts ready to use and easy to modify. Refer to :ref:`use` for instructions on how to use the supplemental software. Combined together both of the packages: cohere and cohere-ui offer a full solution from beamline preprocessing data followed by standard preprocessing data, reconstruction, and beamline visualization. The project was implemented for the Advanced Photon Source beamline 34-IDC and thus the beamline data preparation and data visualization is customized for that specific beamline. The measurements and parameters that were used during the experiment are parsed specifically for that beamline setup. We offer the code, as the community would benefit from it when customizing for own cases.

The cohere is written in Python as this choice offers simplicity of development, and thus the tool can be expanded by community.

A legacy version (1.2.4) written in Python/c++ is still available  in the legacy branch, and available from anaconda but there is no new development in this branch.