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

- beamline:
| optional, beamline where the experiment was conducted. If not configured, the beamline preprocessing and beamline visualization scripts are not available.
| example:
::

    beamline = "aps_34idc"

- multipeak:
| optional, boolean parameter indicating whether it is multi peak case. Defaults to False.
| example:
::

    multipeak = True

- separate_scans:
| optional, deafults to False. In typical scenario the data from all scans in experiment are combined. If specified as separate scans, each scan will be processed separately and will have sub-experiment name containing scan index ex. "scan_9", where 9 is scan index
| example:
::

   separate_scans = False

- separate_scan_ranges:
| optional, deafults to False. In typical scenario the data from all scans in experiment are combined. If specified as separate scan ranges, each scan or scan range in the experiment will be processed separately and will have sub-experiment name containing scan index ex. "scan_9", where 9 is scan index, or "scan_10-15", where 10-15 is the scan range. The scans and scan ranges are defined in main configuration "config" file as scan parameter, and are part of experiment name.
| example:
::

   separate_scan_ranges = True

- auto-data:
| optional, boolean parameter indicating automated data preprocessing, which includes exclusion of outlier scans in multi-scan data and auto calculation of intensity threshold. Defaults to False.
| example:
::

    multipeak = True

- converter_ver:
| mandatory after ver 3.0, if not configured, it will be auto-updated by scripts to match the latest converter version.
| example:
::

    converter_ver = 1
