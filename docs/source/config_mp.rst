========
config_mp
========
- scan:
| mandatory, string type encapsulating scans or ranges of scans containing data for each peak. The scans/scan ranges should be arranged in ascending order.
| example:
::

    scan = "898-913,919-934,940-955,961-976"

- orientations:
| mandatory, a list of lists, each inner list defining the orientation of a peak.
| example:
::

    orientations = [[-1, -1, 1], [0, 0, 2], [1, 1, 1], [2, 0, 0]]

- hkl_in:
| mandatory, list with Miller indices representing the in-plane lattice vector.
| example:
::

    hkl_in = [3.031127677370978, 12.31353345906843, 8.75104158816168]

- hkl_out:
| mandatory, list with Miller indices representing the out-of-plane lattice vector.
| example:
::

    hkl_out = [9.77805918769193, -8.402719849515048, 8.436553021703112]

- twin_plane:
| mandatory, list with Miller indices of the twin plane. If there is not a twin plane, this should be the same as sample_axis
| example:
::

    twin_plane = [1, -1, 1]

- sample_axis:
| mandatory, axis of the sample. The data will be rotated so that the twin plane vector in the lattice frame (hkl) corresponds to this vector in the array frame (xyz).
| example:
::

    sample_axis = [0, 1, 0]

- final_size:
| mandatory, a size in each dimension of the array holding reconstructed object.
| example:
::

    final_size = 180

- mp_max_weight:
| optional, a number between 0 and 1.0 specifying the initial coupling weight assigned to all peaks. if not provided, defaults to 1.0.
| example:
::

    mp_max_weight = 1.0

- mp_taper:
| mandatory, fraction of the total iterations at which to begin tapering the coupling weight. Coupling weight will decrease for all peaks starting at mp_taper times the total number of iterations.
| example:
::

    mp_taper = 0.6

- lattice_size:
| mandatory, lattice parameter of the reconstructing crystal. This is used to define the reciprocal lattice vectors, which are required for projecting to each peak.
| example:
::

    lattice_size = 0.4078

- switch_peak_trigger:
| mandatory, a trigger defining at which iteration to switch the peak
| example:
::

    switch_peak_trigger = [0, 50]
