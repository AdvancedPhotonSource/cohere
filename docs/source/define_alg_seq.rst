.. _formula:

=================
execution formula
=================
| The "config_rec" defines parameters used during reconstruction process.
| Parameters are grouped by features.
| The features are controlled by triggers. If trigger is defined then the feature is active and related parameters are used when processing the feature.
| Triggers and algorithm_sequence parameter define which features are used in each iteration. Trigger can be defined as a single trigger or a list of sub-triggers.
| Algorithm_sequence define the algorithms used in iterations. The algorithm_sequence and triggers create a formula.
| The trigger and algorithm sequence are explained in the sections below.

Trigger
=======
| Trigger concludes at which iteration to apply the associated feature action. Trigger is defined as a list and can be configured for a single iteration, or multiple iterations.
| examples:
| [3] trigger at iteration 3
| [20, 5] trigger starts at iteration 20, repeats every 5 iteration for the rest of run
| [20, 5, 40] trigger starts at iteration 20, repeats every 5 iteration until iteration 40

Sub-Trigger
===========
| For the following features: shrink wrap, lowpass filter, and phc (phase constrain), the trigger can be defined as a list of sub-triggers. In this case the algorithm_sequence should include the sub-triggers in the sequence, for example: 30*ER.SW0, and the trigger should be defined as a list of sub-triggers.
| examples:
| [[5, 1, 100], [0, 3], [5]] three sub-triggers are defined and the rules of the trigger apply for sub-trigger with the difference that the span of iterations in sub-trigger is the same as the algorithm it is attached to. In this example 30*ER.SW0, the start iteration is when the ER begins, and last iteration is when the ER ends.


- algorithm_sequence:
| defines sequence of algorithms applied in each iteration during modulus projection and during modulus. The "*" character means repeat, and the "+" means add to the sequence. The sequence may contain single brackets defining a group that will be repeated by the preceding multiplier. The alphabetic entries: ER, ERpc, HIO, HIOpc define algorithms used in this iteration. The entries will invoke functions as follows: ER definition will invoke 'er' and 'modulus' functions, the ERpc will invoke 'er' and 'pc_modulus', HIO will invoke 'hio' and 'modulus', and HIOpc will invoke 'hio' and 'pc_modulus'. The pc_modulus is implementation of modulus with partial coherence correction. If defining ERpc or HIOpc the pcdi feature must be activated by configuring pc_interval.
| Algorithm sequence can include sub-triggers attached to the algorithm, for example 30*ER.SW0.PHC1.

Formula examples
================
| Algorithm sequence and corresponding trigger configuration
| example1:
::

    algorithm_sequence = "2* (20*ER + 180*HIO) + 2* (20*ERpc + 180*HIOpc) + 20*ERpc"

config_rec:
|    shrink_wrap_trigger = [1, 3]
|    lowpass_trigger = [0, 2, 130]

| In this example the program will run twice 20 iterations with ER algorithm and modulus and 180 iterations with HIO algorithm and modulus
| followed by twice 20 iterations with ER algorithm and partial coherence modulus and 180 iterations with HIO algorithm and partial coherence modulus
| followed by 20 iterations with ER algorithm and partial coherence modulus
| In addition, based on the config_rec, every three iterations starting from first, for the rest of the run, the shrink wrap trigger will be applied
| and every two iterations, starting at the beginning, until iteration 130, the lowpass trigger will be applied.

| example2:
::

    algorithm_sequence = "20*ER.SW0.PHC0 + 180*HIO.SW1.PHC1 + 20*ER.SW2"

config_rec:
|    shrink_wrap_trigger = [[1, 2], [0, 3], [0, 4]]
|    phc_trigger = [[0, 2, 100], [0, 3, 100]]

| In this example the program will run 20 iterations with ER algorithm and modulus. During this 20 iterations the first sub-trigger in shrink_wrap_trigger, defined as [1,2] will be applied, which will start at iteration one and proceed every second iteration. During these iterations first phase modulus sub-trigger from phc_trigger, defined as [0, 2 ,100] will be applied starting at the beginning iteration, repeating every other iteration, and will continue until the 20 ER iterations, even though it was configured for more iterations.
| The ER iterations will be followed by 180 iterations with HIO algorithm and modulus and sub-triggers. During this 180 iterations the second sub-trigger in shrink_wrap_trigger, defined as [0, 3] will be applied, which will start at first HIO iteration and proceed every third iteration. During these iterations second phase modulus sub-trigger from phc_trigger, defined as [0, 3 ,100] will be applied that will start at the beginning iteration, repeat every third iteration, and will continue until the 100 HIO iterations.
| This sequence will be follow by 20 iterations with ER algorithm and modulus and shrink wrap sub-trigger, defined as [0,4].
