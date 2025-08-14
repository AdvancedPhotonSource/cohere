.. _config_instr:

============
config_instr
============
The "config_instr" file defines parameters related to experiment and instrument used during the experiment, such as detector and diffractometer.

beamline aps_1ide
-----------------
All the listed parameters are mandatory for the 1-IDE beamline.

Parameters
==========
- diffractometer:
| Name of diffractometer used in experiment.
| example:
::

    diffractometer = "1ide"

- energy
| example:
::

    energy = 7.2

- vff_r_offset:
| example:
::

    vff_r_offset = -0.107

- vff_eta_offset:
| example:
::

    vff_eta_offset = 0.232

- aero:
| example:
::

    aero = -10.97994

- vff_eta:
| example:
::

    vff_eta = -24.8

- detdist:
| example:
::

    detdist = 6.22ro = -10.97994

- vff_r:
| example:
::

    vff_r = 692.50052

- scanmot:
| example:
::

    scanmot = "aero"

- scanmot_del:
| example:
::

    scanmot_del = 0.004999999999999997

- detector:
| detector name
| example:
::

    detector = "ASI"

beamline aps_34idc
------------------
The diffractometer is mandatory parameter for the 34-IDC beamline. If specfile is defined, the other parameters are parsed from the specfile. They can be overridden when included in this configuration file.

Parameters
==========
- diffractometer:
| Mandatory, name of diffractometer used in experiment
| example:
::

    diffractometer = "34idc"

- specfile:
| Optional (but necessary when the parsed parameters are not provided), specfile recorded when the experiment was conducted.
| example:
::

    specfile = "/path/to/specfile/Staff20.spec"

Parsed parameters
=================
| In a typical scenario at APS 34-idc beamline a spec file is generated during experiment and the parameters listed below are parsed from this file, so the user do not configure them. User may override the parameters.
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

beamline esrf_id01
------------------
Parameters
==========
- diffractometer:
| Mandatory, name of diffractometer used in experiment.
| example:
::

    diffractometer = "id01"

- detector:
| Detector name
| example:
::

    detector = "mpxgaas"

- h5file:
| File of hd5 format containing data and metadata.
| example:
::

    h5file = "path/to/data_file.h5"


beamline Petra3_P10
-------------------
Parameters
==========
- diffractometer:
| Mandatory, name of diffractometer used in experiment.
| Must be defined for the beamline.
| example:
::

    diffractometer = "P10sixc"

- sample:
| Defines sample name that is used in directory structure.
| example:
::

    sample = "MC7_insitu"

- data_dir:
| Directory where experiment data is stored; contains subdirectories related to samples.
| example:
::

    data_dir = "example_data"
