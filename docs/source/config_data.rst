.. _config_dt:

===========
config_data
===========
| The "config_data" file defines parameters needed to format data for phasing. The input is an instrument corrected data file obtained in the beamline preprocessing phase.

Parameters
==========
- data_dir:
| Optional, defaults to <experiment_dir>/preprocessed_data in cohere experiment directory structure.
| Defines directory that contains data.tif file with beamline preprocessed experiment data.
| example:
::

    data_dir = "/path/to/data_dir"

- alien_alg:
| Optional, if not defined, no alien algorithm is used.
| Name of method used to remove aliens. Possible options are: 'block_aliens', 'alien_file', and 'AutoAlien1'. The 'block_aliens' algorithm will zero out defined blocks, 'alien_file' method will use given file as a mask, and 'AutoAlien1' will use auto mechanism to remove aliens. Each of these algorithms require different parameters, explained below.
| example:
::

    alien_alg = "AutoAlien1"

- aliens:
| Needed when the 'block_aliens' method is configured. Used when the data contains regions with intensity produced by interference. The regions are zeroed out. The aliens can be defined as regions, each region defined by coordinates of starting point, and ending point (i.e. [[xb0,yb0,zb0,xe0,ye0,ze0],[xb1,yb1,zb1,xe1,ye1,ze1],...[xbn,ybn,zbn,xen,yen,zen]] ).
| example:
::

    aliens = [[170,220,112,195,245,123], [50,96,10,60,110,20]]

- alien_file:
| Needed when the 'alien_file' method is configured. User can produce a file in npy format that contains zeros and ones, where zero means to set the pixel to zero, and one to leave it.
| example:
::

    alien_file = "/path/to/mask_file/AlienImg.npy"

- AA1_size_threshold:
| Used in the 'AutoAliens1' method. If not given it will default to 0.01.  The AutoAlien1 algorithm will calculate relative sizes of all clusters with respect to the biggest cluster. The clusters with relative size smaller than the given threshold will be possibly deemed aliens. It also depends on asymmetry.
| example:
::

    AA1_size_threshold = 0.01

- AA1_asym_threshold:
| Used in the 'AutoAliens1' method. If not given it will default to 1.75. The AutoAlien1 algorithm will calculate average asymmetry of all clusters. The clusters with average asymmetry greater than the given threshold will be possibly deemed aliens. It also depends on relative size.
| example:
::

    AA1_asym_threshold = 1.75

- AA1_min_pts:
| used in the 'AutoAliens1' method. If not given it will default to 5. Defines minimum non zero points in neighborhood to count the area of data as cluster.
| example:
::

    AA1_min_pts = 5

- AA1_eps:
| Used in the 'AutoAliens1' method. If not given it will default to 1.1. Defines neighborhood Used in the clustering algorithm.
| example:
::

    AA1_eps = 1.1

- AA1_amp_threshold:
| Mandatory in the 'AutoAliens1' method. Used to zero data points below that threshold.
| example:
::

    AA1_amp_threshold = 6 

- AA1_save_arrs
| Used in the 'AutoAliens1' method, optional. If given and set to True multiple results of alien analysis will be saved in files.
| example:
::

    AA1_save_arrs = True 

- AA1_expandcleanedsigma:
| Used in the 'AutoAliens1' method, optional. If given the algorithm will apply last step of cleaning the data using the configured sigma.
| example:
::

    AA1_expandcleanedsigma = 5.0

- auto_intensity_threshold:
| Optional, defaults to False.
| The intensity threshold is calculated programmatically if set to True, otherwise must be provided.
| example:
::

    auto_intensity_threshold = True

- intensity_threshold:
| Mandatory, if auto_intensity_threshold is not set. Intensity values below this value are set to 0. The threshold is applied after removing aliens.
| If auto_data is configured , this value is overridden by calculated value.
| example:
::

    intensity_threshold = 25.0

- crop_pad:
| Optional, a list of numbers defining how to adjust the size at each side of 3D data. If number is positive, the array will be padded. If negative, cropped. The parameters correspond to [x left, x right, y left, y right, z left, z right]. The final dimensions will be adjusted up to the good number for the FFT such as product of powers of 2, 3 or 5.
| example:
::

    crop_pad = [13, 0, -65, -65, -65, -65]

- center_shift:
| Optional, defines offset of max element from the array center.
| example:
::

    center_shift = [0,0,0]

- no_center_max:
| Optional, defaults to False. If False the array maximum is centered, otherwise max is not moved.
| example:
::
    
    no_center_max = False

- binning:
| Optional, a list that defines binning values in respective dimensions, [1,1,1] has no effect.
| example:
::

    binning = [1,1,1]

