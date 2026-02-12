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

- user_roi

| Optional, defines user region of interest within frame. The indices are related to the data already cropped to the area detector roi.

::

     user_roi = [76,64,200,200]

- roi_format

| Mandatory if user_roi is defined, interprets the user_roi entry. Supported values:
| "center_point_dist" : [center_point_x, center_point_y, distance_x, distance_y]
| "start_point_end_point" : [start_point_x, start_point_y, end_point_x, end_point_y]
| "start_point_distance" : [start_point_x, distance_x, start_point_y, distance_y]

::

     roi_format = "start_point_end_point"

- max_crop

| Size of frame cut out around maximum. It will be applied after user_roi if both parameters are defined.

::

    max_crop = [200,200]

- remove_outliers

| Optional, boolean parameter for auto removal of outlier scans in large data sets.

::

    remove_outliers = True

- outliers_scans

| This list is determined automatically when remove_outliers is set to True.

::

    outliers_scans = [78,80]
