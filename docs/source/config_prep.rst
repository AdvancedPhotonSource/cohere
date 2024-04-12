===========                                      
config_prep
===========
| The "config_prep" file defines parameters used during data preparation, when reading data. This file contains parameters used by script written for APS 34-IDC beamline.

Parameters
==========

- data_dir:
| Mandatory for the 34-IDC prep, the directory containing raw data. This directory will have several subdirectories each containing tif format data files. Each subdirectory represents separate scan and is numbered with the scan index.
| example:
::

    data_dir = "/path/to/data/files/ADStaff19-1a"                                             

- darkfield_filename:
| Optional, depending on the detector. Dark field file taken to clean bad pixels or correct background in CCD. Each detector needs to implement this correctly.
| example:
::

    darkfield_filename = "/path/to/darkfield_file/dark.tif"

- whitefield_filename:
| Optional, depending on the detector. White field file taken to correct over saturated pixels. Each detector needs to implement this correctly.
| example:
::

    whitefield_filename = "/path/to/whitefield_file/whitefield.tif"

- roi:
| Typically read from a spec file. Defines the region of a detector that the data has been read from. Needed to determine corrections on the detector. The values in the list (refer to example below) are as follows: x coordinate starting point, x length, y coordinate starting point, y length.
| example:
::

    roi = [0,256,0,256]

- min_files:
| Optional, defines a minimum number of raw data files in the scan directory. If number of files is less than minimum, the directory is not added to the data.
| example:
::

     min_files = 80

- exclude_scans:
| A list containing scan indexes that will be excluded from preparation.
| example:
::

    exclude_scans = [78,81]

- Imult:
| Optional, defaults to the average of the whitefield. A multiplication factor used to renormalize the whitefield correction.
| example:
::

   Imult = 1000000

- outliers_scans:
| This list is determined when running auto-data preprocessing.
| example:
::

    outliers_scans = [78,80]
