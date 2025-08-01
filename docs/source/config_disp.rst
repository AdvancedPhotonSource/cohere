===========
config_disp
===========
| The "config_disp" file defines parameters needed to process visualization of the reconstructed image.  Through the configuration user can obtain various images viewable with Paraview. The reconstructed object, called image is processed for visualization by default. After running beamline_visualizaion.py script user will have image.vts file that contains image as amplitudes and phases (or real and imagnary if configured complex_mode = "ReIm"), support, and optionally unwrapped phases. User can add interpolation to the processing by configuring interpolation_mode. This option will create vts file containing interpolated image. Another optional return is reciprocal space visualization, i.e. phasing data and inverse fourier of that data. User can request it by setting write_recip = True. To view resolution in direct and reciprocal space user can set determine_resolution = "deconv". This will produce vts file containing resolution. If the interpolation is set, the resolution results are automatically saved.

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

- crop_type:
| Optional, Defines how crop is determined. Supported values: "fraction' and "tight". If defined as fraction, the defined fraction of each dimension is cropped around maximum value. The "tight" defines crop being determined by imcrop_margin and imcrop_thresh parameters applied to the image. The extend subarray will be derived from image array by finding points greater than threshold multiplied by maximum value. A margin will be added to each side of the extend array.
| example:
::

    crop_type = "tight"
    crop_type = "fraction"

- crop_fraction:
| Required parameter when crop_type is configured "fraction". Defines size of the cropped array relative to the full image array. The full array is cropped around maximum value.
| example:
::

    crop_fraction = [.5, .5, .5]

- crop_margin:
| Required parameter when crop_type is configured "tight". The margin will be added to each side of the extend array.
| example:
::

    crop_margin = 10

- crop_thresh:
| Required parameter when crop_type is configured "tight". The threshold will determine the extend of the array.
| example:
::

    crop_thresh = 0.5

- complex_mode:
| This mode determines arrays that will be saved in the direct space images file. If mode is "AmpPhase" the "imAmp" and "imPh" arrays will be saved that hold image amplitudes and image phases. if mode is "ReIm" the "imRe" and "imImag" arrays will be saved that hold real values and imaginary values. Defaults to "AmpPhase".
| example:
::

    complex_mode = "AmpPhase"

- interpolation_mode:
| If present the reconstructed object will be interpolated. The parameter defines how the image is interpolated. Supported values: "AmpPhase" and "ReIm". If defined as "AmpPhase" the image amplitudes and image phases are interpolated. If defined as "ReIm" the image real values and imaginary are interpolated, and then the interpolated image amplitudes and image phases are calculated.
| example:
::

    interpolation_mode = "AmpPhase"

- interpolation_resolution:
| Required parameter for interpolation. Supported values: "min_deconv_res", int value, float value, list. If set to "min_deconv_res" the resolution will be determined by including the deconvolution resolution. If defined as integer value the resolution will be set to this value in each dimension. If defined as list, the list will define resolution in corresponding dimension. If set to "min_deconv_res" the resolution capability must be configured by setting the "determine_resolution_type" parameter.
| example:
::

    interpolation_resolution = "min_deconv_res"

- determine_resolution_type:
| If present, the resolution in direct and reciprocal spaces will be found. Supported value: "deconv".
| example:
::

    determine_resolution_type = "deconv"

- resolution_deconv_contrast:
| A fraction less than 0, required when "determine_resolution_type" is set to "deconv".
| example:
::

    resolution_deconv_contrast = 0.25

- write_recip:
| If True the reciprocal_space.vts file will be saved with arrays of phasing data and inverse fourier of that data.
| example:
::

    write_recip = True
