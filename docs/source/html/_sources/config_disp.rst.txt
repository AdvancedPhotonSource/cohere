===========
config_disp
===========
| The "config_disp" file defines parameters needed to process visualization of the reconstructed image.  

Parameters
==========
- results_dir:
| optional, defaults to <experiment_dir>. A directory that has a tree, or leaf with reconstruction results. The results will be used as input to the visualization processing. There could be several results in the given directory tree, and all will be processed concurrently.
| example:
::

    results_dir = "/path/to/phasing_results_dir_tree"

- rampups:                
| optional, upsize when running ramp removal, default is 1. Expect long processing time for greater numbers.
| example:
::

    rampups = 2

- crop:
| optional, defines a fraction of array at the center that will be subject of visualization. Defaults to [1.0, 1.0, 1.0], i.e. the size of the processed array.
| example:
::

    crop = [.5, .5, .5]

- diffractometer:
| mandatory, name of diffractometer used in experiment.
| example:
::

    diffractometer = "34idc"

Parsed parameters
=================
| In a typical scenario at APS 34-idc beamline a spec file is generated during experiment and the parameters listed below are parsed from this file, so the user do not configure them. User may configure the following parameters if spec file is not configured in the main configuration or user wants to override the parsed parameters.
- energy
| example:
::

    energy = 7.2

- delta:
| delta (degrees)
| example:
::

    delta = 30.1

- gamma:
| gamma (degrees)
| example:
::

    gamma = 14.0

- detdist:
| camera distance (mm)
| example:
::

    detdist = 500.0

- theta:
| angular step size
| example:
::

    theta = 0.1999946

- chi:
| example:
::

    chi = 90.0

- phi:
| example:
::

    phi = -5.0

- scanmot:
| example:
::

    scanmot = "th"

- scanmot_del:
| example:
::

    scanmot_del = 0.005

- detector:
| detector name
| example:
::

    detector = "34idcTIM2"
