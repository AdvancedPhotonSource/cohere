============
config_instr
============
| The "config_instr" file defines parameters related to experiment and instrument used during the experiment, such as detector and diffractometer.
| The diffractometer and specfile are mandatory parameters for the 34-IDC beamline. The other parameters are obtained from the specfile. They can be overriden when included in this configuration file.

Parameters
==========
- diffractometer:
| mandatory, name of diffractometer used in experiment.
| example:
::

    diffractometer = "34idc"

- specfile:
| optional (but mandatory for aps_34idc), specfile recorded when the experiment was conducted.
| example:
::

    specfile = "/path/to/specfile/Staff20.spec"

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
