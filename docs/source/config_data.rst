===========
config_data
===========
| The "config_data" file defines parameters needed to format data. The input is a data file obtained in the preparation phase.

Parameters
==========
- data_dir:
| optional, used for specific cases. Driven by the scripts, the experiment directory contains four directories: conf, prep, data, and results. The formatted data is saved in the <experimenr_dir>/data directory. If the data_dir parameter is configured then the data is saved in this directory. Warning: The script running reconstruction expects to read data from <experimenr_dir>/data.
| example:
::

    data_dir = "/path/to/data_dir/data_dir"

- alien_alg:
| optional, name of method used to remove aliens. Possible options are: 'block_aliens', 'alien_file', and 'AutoAlien1'. The 'block_aliens' algorithm will zero out defined blocks, 'alien_file' method will use given file as a mask, and 'AutoAlien1' will use auto mechanism to remove aliens. Each of these algorithms require different parameters, explained below.
| example:
::

    alien_alg = "AutoAlien1"

- aliens:
| Needed when the 'block_aliens' method is configured. Used when the data contains regions with intensity produced by interference. The regions needs to be zeroed out. The aliens can be defined as regions each defined by coordinates of starting point, and ending point (i.e. ((xb0,yb0,zb0,xe0,ye0,ze0),(xb1,yb1,zb1,xe1,ye1,ze1),...(xbn,ybn,zbn,xen,yen,zen))).
| example:
::

    aliens = ((170,220,112,195,245,123), (50,96,10,60,110,20))

- alien_file:
| Needed when the 'alien_file' method is configured. User can produce a file in npy format that contains table of zeros and ones, where zero means to set the pixel to zero, and one to leave it. 
| example:
::

    alien_file = "/path/to/mask_file/AlienImg.npy"

- AA1_size_threshold:
| used in the 'AutoAliens1' method. If not given it will default to 0.01.  The AutoAlien1 algorithm will calculate relative  sizes of all clusters with respect to the biggest cluster. The clusters with relative size smaller than the given threshold will be possibly deemed aliens. It also depends on asymmetry.
| example:
::

    AA1_size_threshold = 0.01

- AA1_asym_threshold:
| used in the 'AutoAliens1' method. If not given it will default to 1.75. The AutoAlien1 algorithm will calculate average asymmetry of all clusters. The clusters with average asymmetry greater than the given threshold will be possibly deemed aliens. It also depends on relative size.
| example:
::

    AA1_asym_threshold = 1.75

- AA1_min_pts:
| used in the 'AutoAliens1' method. If not given it will default to 5. Defines minimum non zero points in neighborhood to count the area of data as cluster.
| example:
::

    AA1_min_pts = 5

- AA1_eps:
| used in the 'AutoAliens1' method. If not given it will default to 1.1. Used in the clustering algorithm.
| example:
::

    AA1_eps = 1.1

- AA1_amp_threshold:
| mandatory in the 'AutoAliens1' method. Used to zero data points below that threshold.
| example:
::

    AA1_amp_threshold = 6 

- AA1_save_arrs
| used in the 'AutoAliens1' method, optional. If given and set to True multiple results of alien analysis will be saved in files.
| example:
::

    AA1_save_arrs = True 

- AA1_expandcleanedsigma:
| used in the 'AutoAliens1' method, optional. If given the algorithm will apply last step of cleaning the data using the configured sigma.
| example:
::

    AA1_expandcleanedsigma = 5.0

- amp_threshold:
| mandatory, min data threshold.  Values below this are set to 0. The threshold is applied after removing aliens.
| example:
::

    amp_threshold = 25.0

- adjust_dimensions:
| optional, a list of number to adjust the size at each side of 3D data. If number is positive, the array will be padded. If negative, cropped. The parameters correspond to (x left, x right, y left, y right, z left, z right) The final dimensions will be adjusted up to the good number for the FFT which also is compatible with opencl supported dimensions powers of 2 or a*2^n, where a is 3, 5, or 9
| example:
::

    adjust_dimensions = (13, 0, -65, -65, -65, -65)

- center_shift:
| optional, enter center shift list the array maximum is centered before binning, and moved according to center_shift, (0,0,0) has no effect
| example:
::

    center_shift = (0,0,0)

- binning:
| optional, a list that defines binning values in respective dimensions, (1,1,1) has no effect
| example:
::

    binning = (1,1,1)

