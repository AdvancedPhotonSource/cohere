===========
config_disp
===========
| The "config_disp" file defines parameters needed to process visualization of the reconstructed image.  

Parameters
==========
- results_dir:
| Optional, defaults to <experiment_dir>/results_phasing. A directory that has a tree, or leaf with reconstruction results. The results will be used as input to the visualization processing. There could be several results in the given directory tree, and all will be processed concurrently.
| example:
::

    results_dir = "/path/to/results_phasing"

- rampups:                
| Optional, upsize when running ramp removal, default is 1. Expect long processing time for greater numbers.
| example:
::

    rampups = 2

- crop:
| Optional, defines a fraction of array at the center that will be subject of visualization. Defaults to [1.0, 1.0, 1.0], i.e. the size of the processed array.
| example:
::

    crop = [.5, .5, .5]

- unwrap:
| Switch, if True the image.vts file will contain unwrapped phase in addition to phase and amplitude.
| example:
::

    unwrap = False

- make_twin:
| A switch to whether visualize twin image.
| example:
::

    make_twin = False
