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
- diffractometer

| Name of diffractometer used in experiment.

::

    diffractometer = "1ide"

- energy

::

    energy = 7.2

- vff_r_offset:

::

    vff_r_offset = -0.107

- vff_eta_offset:

::

    vff_eta_offset = 0.232

- aero:

::

    aero = -10.97994

- vff_eta:

::

    vff_eta = -24.8

- detdist:

::

    detdist = 6.22ro = -10.97994

- vff_r:

::

    vff_r = 692.50052

- scanmot:

::

    scanmot = "aero"

- scanmot_del:

::

    scanmot_del = 0.004999999999999997

- detector

| detector name

::

    detector = "ASI"

beamline aps_34idc
------------------
The diffractometer is mandatory parameter for the 34-IDC beamline. If specfile is defined, the other parameters are parsed from the specfile. They can be overridden when included in this configuration file.

Parameters
==========
- diffractometer

| Mandatory, name of diffractometer used in experiment

::

    diffractometer = "34idc"

- specfile

| Optional (but necessary when the parsed parameters are not provided), specfile recorded when the experiment was conducted.

::

    specfile = "/path/to/specfile/Staff20.spec"

Parsed parameters
=================
| In a typical scenario at APS 34-idc beamline a spec file is generated during experiment and the parameters listed below are parsed from this file, so the user do not configure them. User may override the parameters.

- energy

::

    energy = 7.2

- delta

| delta (degrees)

::

    delta = 30.1

- gamma

| gamma (degrees)

::

    gamma = 14.0

- detdist

| camera distance (mm)

::

    detdist = 500.0

- theta

| angular step size

::

    theta = 0.1999946

- chi

::

    chi = 90.0

- phi

::

    phi = -5.0

- scanmot

::

    scanmot = "th"

- scanmot_del

::

    scanmot_del = 0.005

- detector

| detector name

::

    detector = "34idcTIM2"

beamline esrf_id01
------------------

Parameters
==========
- diffractometer

| Mandatory, name of diffractometer used in experiment.

::

    diffractometer = "id01"

- detector

| Detector name

::

    detector = "mpxgaas"

- h5file

| File of hd5 format containing data and metadata.

::

    h5file = "path/to/data_file.h5"


beamline Petra3_P10
-------------------

Parameters
==========
- diffractometer

| Mandatory, name of diffractometer used in experiment.
| Must be defined for the beamline.

::

    diffractometer = "P10sixc"

- sample

| Defines sample name that is used in directory structure.

::

    sample = "MC7_insitu"

- data_dir

| Directory where experiment data is stored; contains subdirectories related to samples.

::

    data_dir = "example_data"
