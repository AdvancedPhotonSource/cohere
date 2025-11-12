.. _config_prep:

===========
config_prep
===========
| The "config_prep" file defines parameters used during data preprocessing.

Parameters
==========

- min_frames

| Optional, defines a minimum number of raw data files in the scan directory. If number of files is less than minimum, the directory is not processed.

::

     min_frames = 80

- exclude_scans

| A list containing scan indexes that will be excluded from preprocessing process.

::

    exclude_scans = [78,81]

- remove_outliers

| Optional, boolean parameter for auto removal of outlier scans in large data sets.

::

    remove_outliers = True

- outliers_scans

| This list is determined automatically when remove_outliers is set to True.

::

    outliers_scans = [78,80]

- max_crop

| Size of frame cut out around maximum

::

    max_crop = [200,200]
