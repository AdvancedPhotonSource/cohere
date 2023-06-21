===========
config_disp
===========
| The "config_disp" file defines parameters needed to process visualization of the reconstructed image.  

Parameters
==========
- results_dir:
| optional, defaults to <experiment_dir>. A directory that has a tree, or leaf with reconstruction results. The results will be used as input to the visualization processing. There could be several results in the given directory tree, and all will be processed concurrently.
| example:
::

    results_dir = "/path/to/phasing_results_dir_tree"

- rampups:                
| optional, upsize when running ramp removal, default is 1. Expect long processing time for greater numbers.
| example:
::

    rampups = 2

- crop:
| optional, defines a fraction of array at the center that will be subject of visualization. Defaults to [1.0, 1.0, 1.0], i.e. the size of the processed array.
| example:
::

    crop = [.5, .5, .5]
