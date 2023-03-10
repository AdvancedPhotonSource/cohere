=====
cofig
=====
| The "config" is a main configuration defining the experiment and is applicable to all parts of processing: beamline preprocessing, standard preprocessing, phasing, and beamline visualization.

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
| optional (but typically needed), string type encapsulating a single number, a range, or combination of numbers and ranges separated by comma, defining scans that will be read and combined to create data file. This parameter should not be set when running multi peak case.
| example:
::

    scan = "2-7, 10, 15, 20-22"

- specfile:
| optional (but typically needed), specfile recorded when the experiment was conducted.
| example:
::

    specfile = "/path/to/specfile/Staff20.spec"
                                        
- beamline:
| optional, beamline where the experiment was conducted. If not configured, the beamline preprocessing and beamline visualization scripts are not available.
| example:
::

    beamline = "aps_34idc"

- converter_ver:
| optional, if not configured, it will be auto-updated by scripts to match the latest converter version.
| example:
::

    converter_ver = 0

- multipeak:
| optional, boolean parameter indicating whether it is multi peak case. Defaults to False.
| example:
::

    multipeak = True
