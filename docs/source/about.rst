=====
About
=====

The cohere package provides tools for reconstruction of image of a nanoscale structures from data obtained using Bragg Coherent Diffraction Imaging technique.

The reconstruction has very good performance, in particular when utilizing GPU. User has a choice to run on cpu or GPU by choosing the processing library. 
The solution offers parallel processing for fast reconstruction of multiple starting points.

A powerful features that can deliver good reconstruction result offered by the cohere package is genetic algorithm (GA).

The tools are supplemented by another package cohere-ui which contains users scripts ready to use and easy to modify. 
Refer to :ref:`use` for instructions on how to use the supplemental software. Combined together both of the packages: cohere_core and cohere_ui offer a full solution from beamline preprocessing data followed by standard preprocessing data, reconstruction, and beamline visualization. 
The cohere_core project handles standard preprocessing on beamline data that has been corrected, and reconstruction. These processes apply to any beamline. 
The beamline preprocessing, where the raw data is corrected for beamline specific instrument, and beamline postprocessing (visualization) are handled in cohere_ui suplemental project.
Currently cohere_ui supports the following beamlines: aps_1ide, aps_34idc, esrf_id01, Petra3_P10.
Another supplemental package cohere_examples provides examples with configuration files and data for the supported beamlines.

The cohere is written in Python as this choice offers simplicity of development, and thus the tool can be expanded by community.
