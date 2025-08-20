.. _config_rec:

==========
config_rec
==========
| The "config_rec" file defines parameters used during reconstruction process.
|
| Parameters are grouped by trigger and corresponding parameters. There is a group of parameters that are not associated with any trigger, and thus, we call them 'general'.
| If trigger is defined then the feature is active and related parameters are used when processing the feature. Refer to :ref:`formula` page, for explanation of triggers, sub-triggers, and algorithm sequence.

Parameters
==========

General
+++++++
- data_dir

| Optional, defines a directory with file named "data.tif" that contains experiment data file ready for reconstruction. Default is <experiment_dir>/phasing_data.

::

    data_dir = "/path/to/phasing_data"

- save_dir

| Optional, used for specific cases, defualt is <experiment_dir>/results_phasing. Driven by the scripts, the experiment directory contains four directories: conf, preprocessed_data, phasing_data, results_phasing, and results_viz. The  reconstruction results are saved in <experiment_dir>/results_phasing directory. If the save_dir parameter is configured then the reconstruction result is saved in this directory.

::

    save_dir = "/path/to/results_phasing"

- init_guess

| Optional, defines how the initial guess is set. Possible options are: 'random', 'continue', and 'AI_guess'. The choice "random" will generate random guess, "continue" will start from previously saved results, and "AI_guess" will run AI reconstruction that will be an initial guess saved at <experiment_dir>/results_AI. Each of these algorithms require different parameters, explained below. The default is 'random'.

::

    init_guess = "random"

- continue_dir

| Must be defined if the init_guess parameter is set to 'continue'. Directory from which initial guess, initial support are read for reconstruction continuation. If the directory contains multiple subdirectories with initial guesses, a thread will start for each subdirectory.

::

    continue_dir = "/path/to/some_phasing_results_dir"

- AI_trained_model

| Must be defined, if init_guess is "AI_guess". Defines the file in hdf5 format that holds trained model.

::

    AI_trained_model = "/path/to/trained/model/trained_model.hdf5"

- reconstructions

| Optional, default is 1. Number of reconstructions to start with. Typically used when running genetic algorithm.

::

    reconstructions = 5

- processing

| Optional, the library used when running reconstruction. When the auto option is selected the program will use the best performing library that is available, in the following order: cupy, torch, numpy. The cp option will utilize cupy, torch will utilize torch, and np will utilize numpy. Default is auto.

::

    processing = "auto"

- device

| Optional, GPU IDs of the target devices for reconstruction(s) or 'all' if all available GPUs should be used. If not defined, the reconstruction process will run on CPU. For cluster configuration it is defined as dict with hosts names as keys and values as described.

::

    device = [0,1,2,7]
    device = 'all'
    device = {'host1':'all', 'host2':[0,1,2,3,4]}

- algorithm_sequence

| Mandatory, defines sequence of algorithms applied in each iteration during modulus projection and during modulus. The "*" character means repeat, and the "+" means add to the sequence. The sequence may contain single brackets defining a group that will be repeated by the preceding multiplier. The alphabetic entries: ER, ERpc, HIO, HIOpc define algorithms used in this iteration. The entries will invoke functions as follows: ER definition will invoke 'er' and 'modulus' functions, the ERpc will invoke 'er' and 'pc_modulus', HIO will invoke 'hio' and 'modulus', and HIOpc will invoke 'hio' and 'pc_modulus', 'RAAR' will invoke 'raar' and 'modulus', 'SF' will invoke 'sf' and 'modulus'. The pc_modulus is implementation of modulus with partial coherence correction. In second example the sequence contains sub-triggers, explained in  :ref:`formula` page.

::

    algorithm_sequence = "2* (20*SF + 180*HIO) + 2* (20*ERpc + 180*HIOpc) + 20*ERpc"
    algorithm_sequence = "20*ER.LPF0.PHC0 + 180*HIO.LPF1 + 2* (20*ER.SW0 + 180*HIO.SW1) + 20*ER.SW2"

- hio_beta

| Optional, default is .9. A parameter used in hio algorithm.

::

    hio_beta = .9

- initial_support_area

| Optional, defaults to [.5,.5,.5]. The list define dimensions of initial support area. The support area is calculated by multiplying the fractions in this parameter by the corresponding data array dimensions. The support array is centered.

::

    initial_support_area = [.5,.5,.5]

Twin
++++

- twin_trigger

| Defines at which iteration to eliminate "twin", i.e. the image under reconstruction is trimmed by preserving a quadrant across x and y dimensions and zeroing the rest of the array.

::

    twin_trigger = [2]

- twin_halves

| Optional, defines which quadrant of the array is preserved in x and y dimensions, defaults to (0, 0).
| Possible choices: [0, 0], [0, 1], [1, 0], [1,1]

::

    twin_halves = [0, 0]

Shrink wrap
+++++++++++
| Support area is an array that defines region in which the image is meaningful. This area is recalculated at the shrink wrap trigger iteration, shrinking along when the image develops. The calculations employ an algorithm defined here as shrink_wrap_type.

- shrink_wrap_trigger

| Defines when to update support array using the parameters below.
| Alternatively can be defined as list of sub-triggers. If sub-triggers are used, the parameters must be lists as well.

::

    shrink_wrap_trigger = [10, 1]
    shrink_wrap_trigger = [[10, 1],[0,5,100],[0,2]]   # sub-triggers

- shrink_wrap_type

| Mandatory, defines type of shrink wrap. Currently only the "GAUSS" type is supported that applies gaussian filter to find support area.

::

    shrink_wrap_type = "GAUSS"
    shrink_wrap_type = [GAUSS, GAUSS, GAUSS]  # sub-triggers

- shrink_wrap_threshold

| Mandatory, defines a threshold value used in the gaussian filter algorithm.

::

    shrink_wrap_threshold = 0.1
    shrink_wrap_threshold = [0.1, 0.11, .12]  # sub-triggers

- shrink_wrap_gauss_sigma

| Mandatory, defines a sigma value used in the gaussian filter algorithm.

::

    shrink_wrap_gauss_sigma = 1.0
    shrink_wrap_gauss_sigma = [1.0, 1.1, 1.2]  # sub-triggers

Phase constrain
+++++++++++++++
| At the beginning iterations the support area is modified in respect to the phase. Support area will include only points with calculated phase inside of the defined bounds.
| Alternatively can be defined as list of sub-triggers. If sub-triggers are used, the parameters must be lists as well.

- phc_trigger

| Defines when to update support array using the parameters below by applying phase constrain.

::

    phc_trigger = [0, 1, 310]
    phc_trigger = [[0, 1, 310], [0,2]]  # sub-triggers

- phc_phase_min

| Mandatory, defines lower bound phase.

::

    phc_phase_min = -1.57
    phc_phase_min = [-1.5, -1.57]  # sub-triggers

- phc_phase_max

| Mandatory, defines upper bound phase.

::

    phc_phase_max = 1.57
    phc_phase_max = [1.5, 1.57]  # sub-triggers

Partial coherence
+++++++++++++++++
| Partial coherence operation initiates recalculation of coherence of data with respect to the amplitudes in reciprocal space. After coherence array is determined, it is used in convolution operation during modulus in subsequent iteration. The coherence array is updated periodically, as defined by the pc_interval. Partial coherence operation is active if the interval is defined and the algorithm sequence indicates "pc".

- pc_interval

| Defines iteration interval between coherence update.

::

    pc_interval = 50

- pc_type

| Partial coherence algorithm. Currently "LUCY" is supported.

::

    pc_type = "LUCY"

- pc_LUCY_iterations

| Optional, defaults to 20. Defines number of iteration inside LUCY algorithm.

::

    pc_LUCY_iterations = 20

- pc_normalize

| Optional, defaults to True. Internal.

::

    pc_normalize = True

- pc_LUCY_kernel

| Mandatory, coherence array area.

::

    pc_LUCY_kernel = [16, 16, 16]

Lowpass Filter
++++++++++++++
| When active, a lowpass Gaussian filter is applied on data, with iteration dependent sigma calculated by line-spacing the lowpass_filter_range parameter over trigger span iterations. Simultaneously, the Gaussian type of shrink wrap is applied with the reverse sigma. The low resolution trigger is typically configured to be active at the first part of iterations.

- lowpass_filter_trigger

| Defines when to apply lowpass filter operation using the parameters below. Typically the last trigger is configured at half of total iterations.
| Alternatively, it can be defined as list of sub-triggers. If sub-triggers are used, the parameters must be lists as well.

::

    lowpass_filter_trigger = [0, 1, 320]
    lowpass_filter_trigger = [[0, 1], [0, 2, 100]]  # sub-triggers

- lowpass_filter_range

| The range is line-spaced over trigger iterations to form a list of iteration dependent sigmas. If only one number is given, the last sigma will default to 1.

::

    lowpass_filter_range = [.7, 1.0]
    lowpass_filter_range = [[.7, .8], [.8, 1.0]]  # sub-triggers

- lowpass_filter_sw_threshold

| During lowpass iterations a GAUSS type shrink wrap is applied with this threshold ans sigma calculated as reverse of low pass filter.

::

    lowpass_filter_sw_threshold = 2.0
    lowpass_filter_sw_threshold = [2.0, 2.0]  # sub-triggers

averaging
+++++++++
| When this feature is activated the amplitudes of the last several iterations are averaged.

- average_trigger

| Defines when to apply averaging. Negative start means it is offset from the last iteration.

::

    average_trigger = [-65, 1]

progress
++++++++
- progress_trigger

| Defines when to print info on the console. The info includes current iteration and error.

::

    progress_trigger = [0, 20]

live viewing
++++++++++++
| This feature allows for a live view of the amplitude, phase, support, and error as the reconstruction develops. With adaptive multipeak phasing, this will instead show the amplitude, phase, measured diffraction pattern, and adapted diffraction pattern. These are shown using a central slice cropped to half the full array size.

- live_trigger

| Defines when to update the live view.

::

    live_trigger = [0, 10]

GA
++
- ga_generations

| Defines number of generations. When defined, and the number is greater than 1, the genetic algorithm (GA) is activated

::

    ga_generations = 3

- ga_metrics

| Optional, a list of metrics that should be used to rank the reconstruction results for subsequent generations. If not defined, or shorter than number of generations, the metric defaults to "chi".
| If the list contains only one element, it will be used by all generations.
| Supported metrics:
| - 'chi': The last error calculated as norm(rs_amplitudes - data)/norm(data).
|           The smallest 'chi' value is the best.
| - 'sharpness': sum(power(abs(image), 4))
|           The smallest 'sharpness' value is the best.
| - 'summed_phase':  angle(image) - sum(angle(image) * support) / sum(support)
|           where support is calculated with shrink wrap using hardcoded threshold=.2 and sigma=.5
|           The greatest 'summed_phase' value is the best.
| - 'area': sum(support)
|           where support is calculated with shrink wrap using hardcoded threshold=.2 and sigma=.5
|           The greatest 'area' value is the best.

::

    ga_metrics = ["chi", "sharpness", "area"]
    ga_metrics = ["chi"]

- ga_breed_modes

| Optional, a list of breeding modes applied to breed consecutive generation. If not defined, or shorter that number of generations, the mode defaults to "sqrt_ab".
| If the list contains only one element, it will be used by all generations.
| Breeding starts with choosing alpha image. The rest of the images are crossed with alpha. Before the crossing, the image, called beta is aligned with alpha, and phases in both of the arrays are normalized to derive ph_alpha = angle(alpha), and ph_beta = angle(beta)
| Supported modes:
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

::

    ga_breed_modes = ["sqrt_ab", "pixel_switch", "none"]
    ga_breed_modes = ["sqrt_ab"]

- ga_cullings

| Optional, defines how many worst samples to remove in a breeding phase for each generation. If not defined for the generation, the culling defaults to 0.

::

    ga_cullings = [2,1]

- ga_shrink_wrap_thresholds

| Optional, a list of threshold values for each generation. The support is recalculated with this threshold after breeding phase. Defaults to configured value of support_threshold.
| If the list contains only one element, it will be used by all generations.

::

    ga_shrink_wrap_thresholds = [.15, .1]

- ga_shrink_wrap_gauss_sigmas

| Optional, a list of sigma values for each generation. The support is recalculated with this sigma after breeding phase. Defaults to configured value of support_sigma.
| If the list contains only one element, it will be used by all generations.

::

    ga_shrink_wrap_gauss_sigmas = [1.1, 1.0]

- ga_lowpass_filter_sigmas

| Optional, a list of sigmas that will be used in subsequent generations to calculate Gaussian low-pass filter applied it to the data. In the example given below this feature will be used in first two generations.

::

    ga_lowpass_filter_sigmas = [2.0, 1.5]

- ga_gen_pc_start

| Optional, a number indicating at which generation the partial coherence will start to be active. If not defined, and the pc feature is active, it will start at the first generation.

::

    ga_gen_pc_start = 3
