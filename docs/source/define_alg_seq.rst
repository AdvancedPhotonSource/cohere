.. _formula:

=================
execution formula
=================
| The "config_rec" defines parameters used during reconstruction process.
| Parameters are grouped by the function (feature) they are utilized for.
| The features are controlled by triggers. If trigger is defined then the feature is active and related parameters are used when processing the feature.
| Triggers and algorithm_sequence parameter define which features are used in each iteration and which algorithm to apply, creating a formula.
| The trigger and algorithm sequence are explain in sections below.

Algorithm Sequence
==================
| Algorithm sequence
| example1:
::

    algorithm_sequence = "2* (20*ER + 180*HIO) + 2* (20*ERpc + 180*HIOpc) + 20*ERpc"
| example2:
::

    algorithm_sequence = "2* (20*ER.SW0.PHM0 + 180*HIO.SW1.PHM1) + 2* (20*ER.SW0 + 180*HIO.SW2) + 20*ER.SW0"

Trigger
=======
| Trigger can be defined as a single iteration, or multiple iterations.
| examples:
| [3] trigger at iteration 3
| [20, 5] trigger starts at iteration 20, repeats every 5 iteration for the rest of run
| [20, 5, 40] trigger starts at iteration 20, repeats every 5 iteration until iteration 40

Sub-Trigger
===========
| For the following features: shrink wrap, lowpass filter, and phase modulus, the definition


- algorithm_sequence:
| mandatory, defines sequence of algorithms applied in each iteration during modulus projection and during modulus. The "*" charcter means repeat, and the "+" means add to the sequence. The sequence may contain single brackets defining a group that will be repeated by the preceding multiplier. The alphabetic entries: ER, ERpc, HIO, HIOpc define algorithms used in this iteration. The entries will invoke functions as follows: ER definition will invoke 'er' and 'modulus' functions, the ERpc will invoke 'er' and 'pc_modulus', HIO will invoke 'hio' and 'modulus', and HIOpc will invoke 'hio' and 'pc_modulus'. The pc_modulus is implementation of modulus with partial coherence correction. If defining ERpc or HIOpc the pcdi feature must be activated. If not activated, the phasing will use modulus function instead.
| example:
::

    algorithm_sequence = "2* (20*ER + 180*HIO) + 2* (20*ERpc + 180*HIOpc) + 20*ERpc"

