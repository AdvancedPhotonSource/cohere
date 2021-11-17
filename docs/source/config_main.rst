=====
cofig
=====
| The "config" is a main configuration defining the experiment and is applicable to all processing.

Parameters
==========
- working_dir:
| mandatory, root directory of the experiment files
| example:
::
    
    working_dir = "/path/to/the/experiment/"

- experiment_id:
| mandatory, a string identifying the experiment
| example:
::

     experiment_id = "ab"

- scan:
| optional (but typically needed), a single number, a range, or combination of numbers and ranges separated by comma, defining scans that will be read and combined to create data file
| example:
::

    scan = "2-7, 10, 15, 20-22"

- specfile:
| optional (but typically needed), specfile recorded when the experiment was conducted.
| example:
::

    specfile = "/path/to/specfile/Staff20.spec"
                                        
- beamline:
| optional (if not configured, the preparation and visualization scripts are not available), beamline where the experiment was conducted.
| example:
::

    beamline = "aps_34idc"

- converter_ver:
| optional, if not configured, it will be auto-updated by scripts to match the latest converter version.
| example:
::

    converter_ver = 0

