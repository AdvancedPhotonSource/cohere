==========
config_rec
==========
| The "config_rec" defines parameters used during reconstruction process.
| Parameters are grouped by the function (feature) they are utilized for. There is a group of parameters that are not associated with any feature, and thus, we call them 'general'.
| The features are controlled by triggers. If trigger is defined then the feature is active and related parameters are used when processing the feature. The section below explains how to configure a trigger.

Trigger
=======
| Trigger can be defined as a single iteration, or multiple iterations.
| examples:
| (3) trigger at iteration 3
| (20, 5) trigger starts at iteration 20, repeats every 5 iteration for the rest of run
| (20, 5, 40) trigger starts at iteration 20, repeats every 5 iteration until iteration 40
| Triggers can also be a combination of any of the above, ex: ((4), (3, 7, 24), (6,20))
  
Parameters
==========
General
+++++++
- data_dir:
| optional, used for specific cases. Driven by the scripts, the experiment directory contains four directories: conf, prep, data, and results. The  data is read from <experimenr_dir>/data directory. If the data_dir parameter is configured then the data is read from this directory.
| example:
::

    data_dir = "/path/to/data_dir/data_dir"

- save_dir:
| optional, used for specific cases. Driven by the scripts, the experiment directory contains four directories: conf, prep, data, and results. The  reconstruction results are saved in <experimenr_dir>/results directory. If the save_dir parameter is configured then the data is saved in this directory.
| example:
::

    save_dir = "/path/to/save_dir/save_dir"

- cont:
| optional, if set to true the reconstruction starts with previous results stored in continue_dir (next configuration parameter). 
| example:
::

    cont = false

- continue_dir:
| must be defined if the "cont" parameter is set to true. Directory from which initial guess, initial support are read for reconstruction continuation. If the directory contains multiple subdirectories with initial guesses, a thread will start for each subdirectory.
| example:
::

    continue_dir = "/path/to/previous_results_dir/previous_results_dir"

- reconstructions:
| optional, default is 1. Number of reconstructions to start with. Typically used when running genetic algorithm.
| example:
::

    reconstructions = 5

- device:
| optional, IDs of the target devices (GPU) for each reconstruction thread. If not defined, the OS will select the GPU, but the processing will not be concurrent. Ignored when running cpu library.
| example:
::

    device = (0,1,2,7)

- algorithm_sequence:
| mandatory, defines algorithm applied in each iteration during modulus projection by a sequence of lists. The first number in a list is a repeat, followed by lists of pairs, each pair defining algorithm and number of iterations to run the algorithm.
| example:
::

    ((3, ("ER",20), ("HIO", 180)), (1,("ER",20)))

- hio_beta:
| optional, default is .9. A parameter used in hio algorithm.
| example:
::

    hio_beta = .9

Twin
++++

- twin_trigger:
| optional, defines at which iteration to eliminate "twin", i.e. the image under reconstruction is trimmed by preserving a quadrant across x and y dimensions and zeroing the rest of the array.
| example:
::

    twin_trigger = (2)

- twin_halves = (0, 0)
| optional, and only applied when twin_trigger is configured. Defines which quadrant of the array is preserved in x and y dimensions, defaults to (0, 0).
| Possible choices: (0, 0), (0, 1), (1, 0), (1,1)
| example:
::

    twin_halves = (0, 0)

Shrink wrap
+++++++
| Support area is an array that defines region in which the image is meaningful. This area is recalculated at the trigger iteration shrinking along when the image develops. The calculations employ an algorithm defined here as shrink_wrap_type.

- shrink_wrap_trigger:
| defines when to update support array using the parameters below.
| example:
::

    shrink_wrap_trigger = (10, 1)

- shrink_wrap_type:
| optional, defaults to "GAUSS". Currently only "GAUSS" is supported
| example:
::

    shrink_wrap_type = "GAUSS"

- shrink_wrap_threshold:
| optional, defaults to 0.1. A threshold value used in the gauss distribution.
| example:
::

    shrink_wrap_threshold = 0.1

- shrink_wrap_gauss_sigma:
| optional, defaults to 1.0. A sigma value used in the gauss distribution.
| example:
::

    shrink_wrap_gauss_sigma = 1.0

- initial_support_area:
| optional, defaults to (.5,.5,.5). The list define dimensions of initial support area. If the values are fractional, the support area will be calculated by multiplying by the data array dimensions. The support array is centered.
| example:
::

    initial_support_area = (.5,.5,.5)

Phase constrain
+++++++++++++++
| At the beginning iterations the support area is modified in respect to the phase. Support area will include only points with calculated phase intside of the defined bounds.

- phase_support_trigger:
| defines when to update support array using the parameters below by applying phase constrain.
| example:
::

    phase_support_trigger = (0, 1, 310)

- phm_phase_min:
| optional, defaults too -1.57. Defines lower bound phase.
| example:
::

    phm_phase_min = -1.57

- phm_phase_max:
| optional, defaults too 1.57. Defines upper bound phase.
| example:
::

    phm_phase_max = 1.57

Partial coherence
+++++++++++++++++
| Partial coherence triggers recalculation of coherence array for the amplitudes in reciprocal space. After first coherence array is determined, it is used for convolution in subsequent iteration.

- pc_trigger:
| defines when to update coherence using the parameters below.
| example:
::

    pc_trigger = (50, 50)

- pc_type:
| mandatory, partial coherence algorithm. Currently "LUCY" is supported.
| example:
::

    pc_type = "LUCY"

- pc_LUCY_iterations:
| optional, defaults to 20. a number of iteration inside LUCY algorithm.
| example:
::

    pc_LUCY_iterations = 20

- pc_normalize:
| optional, defaults to true. Internal.
| example:
::

    pc_normalize = true

- pc_LUCY_kernel:
| mandatory, coherence array area. 
| example:
::

    pc_LUCY_kernel = (16, 16, 16)

Low resolution
++++++++++++++
| When this feature is activated the data is multiplied by Gaussian distribution magnifying the area with meaningful information and the sigma parameter used in calculation of support is modified gradually. The low resolution trigger is typically configured to be active at the beginning iterations and resume around the mid-run. Thus for the remaining iteration the data used in reconstruction is not modified.
- resolution_trigger:
| defines when to apply low resolution using the parameters below. Typically the last trigger is configured at half of total iterations.
| example:
::

    resolution_trigger = (0, 1, 320)

- lowpass_filter_sw_sigma_range:
| used when applying low resolution to replace support sigma at low resolution iterations. The sigmas are linespaced across the defined range for low resolution iterations. If only one number given, the last sigma will default to support_sigma.
| example:
::

    lowpass_filter_sw_sigma_range = (2.0)

- lowpass_filter_range:
| used when applying low resolution to calculate data gauss multiplier. The det values are linespaced for low resolution iterations from first value to last.  The multiplier array is a gauss distribution calculated with sigma of linespaced det across the low resolution iterations. If only one number is given, the last det will default to 1.
| example:
::

    lowpass_filter_range = (.7)

averaging
+++++++++
| When this feature is activated the amplitudes of the last several iterations are averaged.
- average_trigger:
| defines when to apply averaging. Negative start means it is offset from the last iteration
| example:
::

    average_trigger = (-65, 1)

progress
++++++++
- progress_trigger:
| defines when to print info on the console. The info includes current iteration and error
| example:
::

    progress_trigger = (0, 20)

GA
++
- ga_generations:
| optional, number of generations. When defined, and the number is greater than 1, the genetic algorithm (GA) is activated
| example:
::

    ga_generations = 3

- ga_metrics:
| optional, a list of metrics that should be used to rank the reconstruction results for subsequent generations. If not defined, or shorter that number of generations, the metric defaults to "chi".
| supported: "chi", "sharpness", "summed_phase", "area"
| example:
::

    ga_metrics = ("chi", "sharpness", "area")

- ga_breed_modes:
| optional, a list of breeding modes applied to breed consecutive generation. If not defined, or shorter that number of generations, the mode defaults to "sqrt_ab".
| supported: "none", "sqrt_ab", "dsqrt", "pixel_switch", "b_pa", "2ab_a_b", "2a_b_pa", "sqrt_ab_pa", "sqrt_ab_pa_recip", "sqrt_ab_recip", "max_ab", "max_ab_pa", "min_ab_pa", "avg_ab", "avg_ab_pa"
| example:
::

    ga_breed_modes = ("sqrt_ab", "dsqrt", "none")

- ga_cullings:
| optional, defines how many worst samples to remove in a breeding phase for each generation. If not defined for the generation, the culling defaults to 0.
| example:
::

    ga_cullings = (2,1)

- ga_shrink_wrap_thresholds:
| optional, a list of threshold values for each generation. The support is recalculated with this threshold after breeding phase. Defaults to configured value of support_threshold. 
| example:
::

    ga_shrink_wrap_thresholds = (.15, .1)

- ga_shrink_wrap_gauss_sigmas:
| optional, a list of sigma values for each generation. The support is recalculated with this sigma after breeding phase. Defaults to configured value of support_sigma. 
| example:
::

    ga_shrink_wrap_gauss_sigmas = (1.1, 1.0)

- ga_lowpass_filter_sigmas:
| optional, a list of sigmas that will be used in subsequent generations to calculate Gauss distribution in the space defined by the size of the data and applied it to the data. In the example given below this feature will be used in first two generations.
| example:
::

    ga_lowpass_filter_sigmas = (2.0, 1.5)

- ga_gen_pc_start:
| optional, a number indicating at which generation the pcdi feature will start to be active. If not defined, and the pcdi feature is active, it will start at the first generation.
| example:
::

    ga_gen_pc_start = 3
