.. _config_instr:

============
config_instr
============
The "config_instr" file defines parameters related to experiment and instrument used during the experiment, such as detector and diffractometer.

.. _hbb:

common parameters
-----------------
Some detectors may need to remove horizontal bound background. If the detector name is included in det_bound_background list the following parameters may apply:

- remove_band_background

| Boolean switch to determine whether horizontal bound background should be removed.

::

    remove_band_background = True

- rbb_smooth_sigma

| Sigma parameter used in calculations. The optimal value is set for the detector but can be overridden.

::

    rbb_smooth_sigma = 50

- rbb_robust

| A boolean switch to decide a method used in calculation. If robust, a median of data is used, otherwise mean.

::

    rbb_robust = True

beamline aps_1ide
-----------------
All the listed parameters are mandatory for the 1-IDE beamline.

Parameters
==========
- diffractometer

| Name of diffractometer used in experiment.

::

    diffractometer = "1ide"

- specfile

| Optional (but necessary when the parsed parameters are not provided), specfile recorded when the experiment was conducted.

::

    specfile = "/path/to/specfile/aspecfile.spec"

- data_dir

| directory where the raw experiment data is stored

::

    data_dir = "/path/to/data"

- darkfield_filename

| File with detector background noise

::

    darkfield_filename = "/path/to/darkfieldfile/darkfield.tif"

- whitefield_filename

| File with detector whitefield data

::

    whitefield_filename = "/path/to/whitefieldfile/whitwfield.tif"

- Imult

| Optional, value to use when correcting with whitefield.

::

    Imult = 100000

- detector

| Detector name

::

    detector = "ASI"

- energy

::

    energy = 7.2

- detdist:

| Detector distance in mm

::

    detdist = 6220.0

- vff_eta_offset:

| Offset along vff_eta in m

::

    vff_eta_offset = 0.232

- vff_r_offset:

| Offset along vff_r in m

::

    vff_r_offset = -0.107

Parsed parameters
=================
| In a typical scenario at APS 1-ide the parameters listed below are parsed from spec file, so the user do not configure them. User may override the parameters.

- aero:

| aero motor position

::

    aero = -10.97994

- vff_eta:

| Distance along vff_eta in m

::

    vff_eta = -24.8

- vff_r:

| Distance along vff_r in mm

::

    vff_r = 692.50052

- scanmot:

::

    scanmot = "aero"

- scanmot_del:

::

    scanmot_del = 0.004999999999999997


beamline aps_7idd
-----------------
All the listed parameters are mandatory for the 7-IDD beamline.

Parameters
==========
- diffractometer

| Name of diffractometer used in experiment.

::

    diffractometer = "7iddrobot"

- specfile

| Optional (but necessary when the parsed parameters are not provided), specfile recorded when the experiment was conducted.

::

    specfile = "/path/to/specfile/aspecfile.spec"

- data_dir

| directory where the raw experiment data is stored

::

    data_dir = "/path/to/data"

- darkfield_filename

| File with detector background noise

::

    darkfield_filename = "/path/to/darkfieldfile/darkfield.tif"

- whitefield_filename

| File with detector whitefield data

::

    whitefield_filename = "/path/to/whitefieldfile/whitwfield.tif"

- Imult

| Optional, value to use when correcting with whitefield.

::

    Imult = 100000

Parsed parameters
=================
| In a typical scenario at APS 7-idd the parameters listed below are parsed from spec file, so the user do not configure them. User may override the parameters.

- energy

::

    energy = 7.2

- yaw:

| yaw motor position in deg

::

    yaw = 26.5

- pitch:

| pitch motor position in deg

::

    pitch = 0.0

- radius:

| Detector distance in mm

::

    radius = 990.0

- wedge:

| tilt angle set to 10 degrees

::

    wedge = 10.0

- chi:

| chi motor position in deg

::

    chi = 4.4

- th:

| th motor position in deg

::

    th = -0.2225

- phi:

| phi motor position in deg

::

    phi = 14.77

- scanmot:

::

    scanmot = "aero"

- scanmot_del:

::

    scanmot_del = 0.004999999999999997

- detector

| Detector name

::

    detector = "7iddrobot"


beamline aps_20ide
-----------------
All the listed parameters are mandatory for the 20-IDE beamline.

Parameters
==========
- diffractometer

| Name of diffractometer used in experiment.

::

    diffractometer = "7iddrobot"

- data_dir

| directory where the raw experiment data is stored

::

    data_dir = "/path/to/data"

- darkfield_filename

| File with detector background noise

::

    darkfield_filename = "/path/to/darkfieldfile/darkfield.tif"

- whitefield_filename

| File with detector whitefield data

::

    whitefield_filename = "/path/to/whitefieldfile/whitwfield.tif"

- Imult

| Optional, value to use when correcting with whitefield.

::

    Imult = 100000

- detector

| Detector name

::

    detector = "ASI"

| If BSE detector was used, additional parameters used in removal of horizontal bound background are available. Refer to :ref:`hbb`.

Parsed parameters
=================
| In a typical scenario at APS 20-ide the parameters listed below are parsed from h5 data file, so the user do not configure them. User may override the parameters.

- energy

::

    energy = 7.2

- DetX:

| DetX motor position in mm

::

    DetX = 26.5

- DetY:

| DetY motor position in mm

::

    DetY = 0.0

- DetZ:

| Detector distance in mm

::

    DetZ = 990.0

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

- data_dir

| directory where the raw experiment data is stored

::

    data_dir = "/path/to/data"

- darkfield_filename

| File with detector background noise

::

    darkfield_filename = "/path/to/darkfieldfile/darkfield.tif"

- whitefield_filename

| File with detector whitefield data

::

    whitefield_filename = "/path/to/whitefieldfile/whitwfield.tif"

- Imult

| Optional, value to use when correcting with whitefield.

::

    Imult = 100000

Parsed parameters
=================
| In a typical scenario at APS 34-idc beamline a spec file is generated during experiment and the parameters listed below are parsed from this file, so the user do not configure them. User may override the parameters.

- det_roi

| detector area region of interest, defined as follows: [start x, distance x, start y, distance y]

::

    det_roi = [0, 256, 0, 256]

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

::

    diffractometer = "P10sixc"

- data_dir

| Directory where experiment data is stored; contains subdirectories related to samples.

::

    data_dir = "path/to/data"

- sample

| Defines sample name that is used in directory structure.

::

    sample = "MC7_insitu"

- detector_module

| Index defining part of area detector that was used in experiment.

::

    detector_module = 6

Parsed parameters
=================
| In a typical scenario the following metadata is parsed parsed, so the user do not configure them. User may override the parameters.

- energy

::

    energy = 7.2

- del

| delta axis posision (degrees)

::

    delta = 30.1

- gam

| gamma axis posision in (degrees)

::

    gamma = 14.0

- detdist

| camera distance (mm)

::

    detdist = 500.0

- mu

| mu motor position

::

    mu = 0.0

- om

| om motor position

::

    om = 0.1999946

- chi

| chi motor posision in deg

::

    chi = 90.0

- phi

| phi motor posision in deg

| phi detector axis posision in deg

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

    detector = "e4m"
