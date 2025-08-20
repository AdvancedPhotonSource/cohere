.. _formula:

=================
Execution formula
=================
| The "config_rec" file defines parameters used during reconstruction process.
| The execution sequence is determined by algorithm_sequence parameter, triggers, and sub-triggers.
| Algorithm sequence defines which projection algorithm is executed in each iteration. Cohere supports the following projection algorithms: ER (error reduction), HIO (hybrid input-output), SF (solvent flipping), and RAAR (relaxed averaged alternating reflectors). The algorithms are described in this publication: https://pubs.aip.org/aip/rsi/article/78/1/011301/349838/Invited-Article-A-unified-evaluation-of-iterative.
| Triggers define iterations at which the corresponding triggered operations are active.
| The triggered operations use applicable parameters that are grouped by the corresponding trigger.
| Sub-trigger is a trigger applied for part of iterations, associated with the projection algorithm.
| The trigger, sub-trigger, and algorithm sequence are explained in the sections below.

Trigger
=======
| Triggers are defined in config_req, for example: shrink_wrap_trigger = [10, 5]. Refer to :ref:`config_rec` for documentation on all available triggers.
| Trigger defines at which iteration to apply the associated trigger action. Trigger is defined as a list and can be configured for a single iteration, or multiple iterations.
| examples:
| [3] trigger at iteration 3
| [20, 5] trigger starts at iteration 20, repeats every 5 iteration for the rest of run.
| [20, 5, 40] trigger starts at iteration 20, repeats every 5 iteration until iteration 40.

Sub-Trigger
===========
| For sub-trigger configuration, the trigger is a list of triggers, i.e. a list of lists. Each internal list is a sub-trigger.
| The following features: shrink wrap (SW), lowpass filter (LPF), and phase constrain (PHC) can be defined as sub-triggers. The literals listed in parenthesis are used to define the sub-triggers in algorithm_sequence.
| For example a shrink_wrap_trigger = [[5, 1, 100], [0, 3], [5]] defines sub-triggers: SW0, SW1, SW2 respectively.
| Since the sub-triggers are associated with algorithm sequence, they must be attached to the projection algorithm in algorithm_sequence parameter, for example: 20*ER.SW0 + 180*HIO.SW1 + 20*ER.SW2 + 180*HIO.SW1 +20*ER.SW0.
| In this case the first sub-trigger SW0 will be applied during the first 20 iterations, then during next 180 iterations the trigger SW1 is applied, and so on.

Algorithm Sequence
==================
| This parameter defines sequence of algorithms applied in each iteration during modulus projection and during modulus. The "*" character means repeat, and the "+" means add to the sequence. The sequence may contain single brackets defining a group that will be repeated by the preceding multiplier. The alphabetic entries: ER, ERpc, HIO, HIOpc, RAAR, SF  define algorithms used in this iteration. The entries will invoke functions as follows: ER definition will invoke 'er' and 'modulus' functions, the ERpc will invoke 'er' and 'pc_modulus', HIO will invoke 'hio' and 'modulus', and HIOpc will invoke 'hio' and 'pc_modulus', 'RAAR' will invoke 'raar' and modulus, and 'SF' will invoke 'sf' and modulus. The pc_modulus is implementation of modulus with partial coherence correction. If defining ERpc or HIOpc the pcdi (partial coherence) must be activated by configuring pc_interval and pcdi parameters.
| Algorithm sequence can include sub-triggers attached to the algorithm, as described in Sub-Trigger section.

Formula examples
================
| Algorithm sequence and corresponding trigger configuration

| example 1
::

    algorithm_sequence = "2* (20*ER + 180*HIO) + 2* (20*ERpc + 180*HIOpc) + 20*ERpc"

config_rec:
| shrink_wrap_trigger = [1, 3]
| lowpass_trigger = [0, 2, 130]

| In this example the program will run:
| twice 20 iterations with ER algorithm and modulus and 180 iterations with HIO algorithm and modulus,
| followed by twice 20 iterations with ER algorithm and partial coherence modulus and 180 iterations with HIO algorithm and partial coherence modulus,
| followed by 20 iterations with ER algorithm and partial coherence modulus.
| In addition, based on the triggers configuration, every three iterations starting from the first, for the rest of the run, the shrink wrap operation will be applied,
| and every two iterations, starting at the beginning, until iteration 130, the lowpass operation will be applied.

| example 2
::

    algorithm_sequence = "20*ER.SW0.PHC0 + 180*HIO.SW1.PHC1 + 20*ER.SW2"

config_rec:
| shrink_wrap_trigger = [[1, 2], [0, 3], [0, 4]]
| phc_trigger = [[0, 2, 100], [0, 3, 100]]

| In this example the program will run:
| 20 iterations with ER algorithm and modulus. During this 20 iterations the first sub-trigger in shrink_wrap_trigger, defined as [1,2] will be applied, which will start at iteration one and proceed every second iteration. During these iterations first phase constrain sub-trigger from phc_trigger, defined as [0, 2 ,100] will be applied starting at the beginning iteration, repeating every other iteration, and will continue until the 20 ER iterations, even though it was configured for more iterations.
| The ER iterations will be followed by 180 iterations with HIO algorithm and modulus and sub-triggers. During this 180 iterations the second sub-trigger in shrink_wrap_trigger, defined as [0, 3] will be applied, which will start at first HIO iteration and proceed every third iteration. During these iterations the second phase constrain sub-trigger from phc_trigger, defined as [0, 3 ,100] will be applied that will start at the beginning iteration, repeat every third iteration, and will continue until the 100 HIO iterations.
| This sequence will be follow by 20 iterations with ER algorithm and modulus and shrink wrap sub-trigger, defined as [0,4].
