.. _config_rec:

==========
config_rec
==========
| The "config_rec" defines parameters used during reconstruction process.
| Parameters are grouped by function (feature) they are utilized for. There is a group of parameters that are not associated with any feature, and thus, we call them 'general'.
| The features are controlled by triggers. If trigger is defined then the feature is active and related parameters are used when processing the feature. Refer to :ref:`formula` page, for explanation of triggers, sub-triggers, and algorithm sequence.

Parameters
==========
General
+++++++
- data_dir:
| optional, used for specific cases, default is <experiment_dir>/phasing_data. Driven by the scripts, the experiment directory contains four directories: conf, prep, data, and results. The standard preprocessed file, named data.tif should be in data_dir when starting reconstruction. If the data_dir parameter is configured then the data is read from this directory, otherwise from <experiment_dir>/data. This parameter is only accepted when running script from a command line. When using GUI, the data_dir is always default.
| example:
::

    data_dir = "/path/to/data_dir/data_dir"

- save_dir:
| optional, used for specific cases, defualt is <experiment_dir>/results. Driven by the scripts, the experiment directory contains four directories: conf, preprocessed_data, phasing_data, results_phasing, and results_viz. The  reconstruction results are saved in <experiment_dir>/results_phasing directory. If the save_dir parameter is configured then the reconstruction result is saved in this directory.
| example:
::

    save_dir = "/path/to/save_dir/save_dir"

- init_guess:
| optional, defines how the initial guess is set. Possible options are: 'random', 'continue', and 'AI_guess'. The choice "random" will generate random guess, "continue" will start from previously saved results, and "AI_guess" will run AI reconstruction that will be an initial guess saved at <experiment_dir>/results_AI. Each of these algorithms require different parameters, explained below. The default is 'random'.
| example:
::

    init_guess = "random"

- continue_dir:
| must be defined if the init_guess parameter is set to 'continue'. Directory from which initial guess, initial support are read for reconstruction continuation. If the directory contains multiple subdirectories with initial guesses, a thread will start for each subdirectory.
| example:
::

    continue_dir = "/path/to/previous_results_dir/some_phasing_results_dir"

- AI_trained_model:
| must be defined, if init_guess is "AI_guess". defines the file in hdf5 format that holds trained model.
| example:
::

    AI_trained_model = "/path/to/trained/model/trained_model.hdf5"

- reconstructions:
| optional, default is 1. Number of reconstructions to start with. Typically used when running genetic algorithm.
| example:
::

    reconstructions = 5

- processing:
| optional, the library used when running reconstruction. When the auto option is selected the program will use the best performing library that is available, in the following order: cupy, numpy. The cp option will utilize cupy, and np will utilize numpy, and af will leave selection to arrayfire. Default is auto.
| example:
::

    processing = "auto"

- device:
| optional, IDs of the target devices (GPU) for each reconstruction thread. If not defined, the OS will select the GPU, but the processing will not be concurrent. Ignored when running cpu library.
| example:
::

    device = [0,1,2,7]

- algorithm_sequence:
| mandatory, defines sequence of algorithms applied in each iteration during modulus projection and during modulus. The "*" charcter means repeat, and the "+" means add to the sequence. The sequence may contain single brackets defining a group that will be repeated by the preceding multiplier. The alphabetic entries: ER, ERpc, HIO, HIOpc define algorithms used in this iteration. The entries will invoke functions as follows: ER definition will invoke 'er' and 'modulus' functions, the ERpc will invoke 'er' and 'pc_modulus', HIO will invoke 'hio' and 'modulus', and HIOpc will invoke 'hio' and 'pc_modulus'. The pc_modulus is implementation of modulus with partial coherence correction. In second example the sequence contains subtriggers, explained in  :ref:`formula` page.
| examples:
::

    algorithm_sequence = "2* (20*ER + 180*HIO) + 2* (20*ERpc + 180*HIOpc) + 20*ERpc"
    algorithm_sequence = "20*ER.LPF0.PHM0 + 180*HIO.LPF1 + 2* (20*ER.SW0 + 180*HIO.SW1) + 20*ER.SW2"

- hio_beta:
| optional, default is .9. A parameter used in hio algorithm.
| example:
::

    hio_beta = .9

- initial_support_area:
| optional, defaults to [.5,.5,.5]. The list define dimensions of initial support area. If the values are fractional, the support area will be calculated by multiplying by the data array dimensions. The support array is centered.
| example:
::

    initial_support_area = [.5,.5,.5]

Twin
++++

- twin_trigger:
| optional, defines at which iteration to eliminate "twin", i.e. the image under reconstruction is trimmed by preserving a quadrant across x and y dimensions and zeroing the rest of the array.
| example:
::

    twin_trigger = [2]

- twin_halves = [0, 0]
| optional, and only applied when twin_trigger is configured. Defines which quadrant of the array is preserved in x and y dimensions, defaults to (0, 0).
| Possible choices: [0, 0], [0, 1], [1, 0], [1,1]
| example:
::

    twin_halves = [0, 0]

Shrink wrap
+++++++++++
| Support area is an array that defines region in which the image is meaningful. This area is recalculated at the trigger iteration shrinking along when the image develops. The calculations employ an algorithm defined here as shrink_wrap_type.

- shrink_wrap_trigger:
| defines when to update support array using the parameters below.
| alternatively can be defined as list of sub-triggers. If sub-triggers are used, the parameters must be lists as well.
| examples:
::

    shrink_wrap_trigger = [10, 1]
    shrink_wrap_trigger = [[10, 1],[0,5,100],[0,2]]   # sub-triggers

- shrink_wrap_type:
| optional, defaults to "GAUSS" which applies gaussian filter. Currently only "GAUSS" is supported.
| Currently only the GAUSS type is supported.
| examples:
::

    shrink_wrap_type = "GAUSS"
    shrink_wrap_type = [GAUSS, GAUSS, GAUSS]  # sub-triggers

- shrink_wrap_threshold:
| optional, defaults to 0.1. A threshold value used in the gaussian filter algorithm.
| examples:
::

    shrink_wrap_threshold = 0.1
    shrink_wrap_threshold = [0.1, 0.11, .12]  # sub-triggers

- shrink_wrap_gauss_sigma:
| optional, defaults to 1.0. A sigma value used in the gaussian filter algorithm.
| examples:
::

    shrink_wrap_gauss_sigma = 1.0
    shrink_wrap_gauss_sigma = [1.0, 1.1, 1.2]  # sub-triggers

Phase constrain
+++++++++++++++
| At the beginning iterations the support area is modified in respect to the phase. Support area will include only points with calculated phase intside of the defined bounds.
| Alternatively can be defined as list of sub-triggers. If sub-triggers are used, the parameters must be lists as well.

- phase_support_trigger:
| defines when to update support array using the parameters below by applying phase constrain.
| examples:
::

    phase_support_trigger = [0, 1, 310]
    phase_support_trigger = [[0, 1, 310], [0,2]]  # sub-triggers

- phm_phase_min:
| optional, defaults too -1.57. Defines lower bound phase.
| example:
::

    phm_phase_min = -1.57
    phm_phase_min = [-1.5, -1.57]  # sub-triggers

- phm_phase_max:
| optional, defaults too 1.57. Defines upper bound phase.
| examples:
::

    phm_phase_max = 1.57
    phm_phase_max = [1.5, 1.57]  # sub-triggers

Partial coherence
+++++++++++++++++
| Partial coherence triggers recalculation of coherence array for the amplitudes in reciprocal space. After coherence array is determined, it is used for convolution in subsequent iteration. The coherence array is updated as defined by the pc_interval. Partial coherence feature is active if the interval is defined and the algorithm sequence contains algorithm with partial coherence.

- pc_interval:
| defines iteration interval between coherence update.
| example:
::

    pc_interval = 50

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
| optional, defaults to True. Internal.
| example:
::

    pc_normalize = True

- pc_LUCY_kernel:
| mandatory, coherence array area. 
| example:
::

    pc_LUCY_kernel = [16, 16, 16]

Lowpass Filter
++++++++++++++
| When this feature is activated a lowpass Gaussian filter is applied on data with sigma being line-spaced over the range parameter. Simultaneously, the Gaussian type of shrink wrap is applied with the reverse sigma. The low resolution trigger is typically configured to be active at the first part of iterations and resume around the mid-run.
- lowpass_filter_trigger:
| defines when to apply low resolution using the parameters below. Typically the last trigger is configured at half of total iterations.
| Alternatively can be defined as list of sub-triggers. If sub-triggers are used, the parameters must be lists as well.
| examples:
::

    lowpass_filter_trigger = [0, 1, 320]
    lowpass_filter_trigger = [[0, 1], [0, 2, 100]]  # sub-triggers

- lowpass_filter_range:
| The range is linespaced over iterations to form a list of sigmas. The sigmas are used to apply gaussian lowpass filter over data. The inverse of sigma is used in shrink wrap.ue to last. If only one number is given, the last det will default to 1.
| examples:
::

    lowpass_filter_range = [.7, 1.0]
    lowpass_filter_range = [[.7, .8], [.8, 1.0]]  # sub-triggers

- lowpass_filter_sw_threshold:
| during lowpass iterations a GAUSS type shrink wrap is applied with this threshold ans sigma calculated as reverse of low pass filter.
| examples:
::

    lowpass_filter_sw_threshold = 2.0
    lowpass_filter_sw_threshold = [2.0, 2.0]  # sub-triggers

averaging
+++++++++
| When this feature is activated the amplitudes of the last several iterations are averaged.
- average_trigger:
| defines when to apply averaging. Negative start means it is offset from the last iteration
| example:
::

    average_trigger = [-65, 1]

progress
++++++++
- progress_trigger:
| defines when to print info on the console. The info includes current iteration and error
| example:
::

    progress_trigger = [0, 20]

GA
++
- ga_generations:
| optional, number of generations. When defined, and the number is greater than 1, the genetic algorithm (GA) is activated
| example:
::

    ga_generations = 3

- ga_metrics:
| optional, a list of metrics that should be used to rank the reconstruction results for subsequent generations. If not defined, or shorter than number of generations, the metric defaults to "chi".
| supported:
| - 'chi': the last error calculated as norm(rs_amplitudes - data)/norm(data)
|           The smallest 'chi' value is the best.
| - 'sharpness': sum(power(abs(image), 4))
|           The smallest 'sharpness' value is the best.
| - 'summed_phase':  angle(image) - sum(angle(image) * support) / sum(support)
|           where support is calculated with shrink wrap using hardcoded threshold=.2 and sigma=.5
|           The greatest 'summed_phase' value is the best.
| - 'area': sum(support)
|           where support is calculated with shrink wrap using hardcoded threshold=.2 and sigma=.5
|           The greatest 'area' value is the best.
| example:
::

    ga_metrics = ["chi", "sharpness", "area"]

- ga_breed_modes:
| optional, a list of breeding modes applied to breed consecutive generation. If not defined, or shorter that number of generations, the mode defaults to "sqrt_ab".
| Breeding starts with choosing alpha image. The rest of the images are crossed with alpha. Before the crossing, the image, called beta is aligned with alpha, and phases in both of the arrays are normalized to derive ph_alpha = angle(alpha), and ph_beta = angle(beta)
| supported:
| - 'sqrt_ab': sqrt(abs(alpha) * abs(beta)) * exp(0.5j * (ph_beta + ph_alpha))
| - 'pixel_switch': where((cond > 0.5), beta, alpha); cond = random(shape(beta))
| - 'b_pa': abs(beta) * exp(1j * (ph_alpha))
| - '2ab_a_b': 2 * (beta * alpha) / (beta + alpha)
| - '2a_b_pa': (2 * abs(alpha) - abs(beta)) * exp(1j * ph_alpha)
| - 'sqrt_ab_pa': sqrt(abs(alpha) * abs(beta)) * exp(1j * ph_alpha)
| - 'sqrt_ab_recip': fftshift(ifft(fftshift(temp))), where temp is calculated below
|                      t1 = fftshift(fft(fftshift(beta)))
|                      t2 = fftshift(fft(fftshift(alpha)))
|                      temp = sqrt(abs(t1)*abs(t2))*exp(.5j*angle(t1))*exp(.5j*angle(t2))
| - 'max_ab': max(abs(alpha), abs(beta)) * exp(.5j * (ph_beta + ph_alpha))
| - 'max_ab_pa': max(abs(alpha), abs(beta)) * exp(1j * ph_alpha)
| - 'avg_ab': 0.5 * (alpha + beta)
| - 'avg_ab_pa: 0.5 * (abs(alpha) + abs(beta)) * exp(1j * (ph_alpha))
| example:
::

    ga_breed_modes = ["sqrt_ab", "pixel_switch", "none"]

- ga_cullings:
| optional, defines how many worst samples to remove in a breeding phase for each generation. If not defined for the generation, the culling defaults to 0.
| example:
::

    ga_cullings = [2,1]

- ga_shrink_wrap_thresholds:
| optional, a list of threshold values for each generation. The support is recalculated with this threshold after breeding phase. Defaults to configured value of support_threshold. 
| example:
::

    ga_shrink_wrap_thresholds = [.15, .1]

- ga_shrink_wrap_gauss_sigmas:
| optional, a list of sigma values for each generation. The support is recalculated with this sigma after breeding phase. Defaults to configured value of support_sigma. 
| example:
::

    ga_shrink_wrap_gauss_sigmas = [1.1, 1.0]

- ga_lowpass_filter_sigmas:
| optional, a list of sigmas that will be used in subsequent generations to calculate Gauss distribution in the space defined by the size of the data and applied it to the data. In the example given below this feature will be used in first two generations.
| example:
::

    ga_lowpass_filter_sigmas = [2.0, 1.5]

- ga_gen_pc_start:
| optional, a number indicating at which generation the pcdi feature will start to be active. If not defined, and the pcdi feature is active, it will start at the first generation.
| example:
::

    ga_gen_pc_start = 3
