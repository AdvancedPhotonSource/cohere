========
config_mp
========
- scan:
| Mandatory, string type encapsulating scans or ranges of scans containing data for each peak. The scans/scan ranges should be arranged in ascending order.
| example:
::

    scan = "898-913,919-934,940-955,961-976"

- orientations:
| Mandatory, a list of lists, each inner list defining the orientation of a peak.
| example:
::

    orientations = [[-1, -1, 1], [0, 0, 2], [1, 1, 1], [2, 0, 0]]

- hkl_in:
| Mandatory, list with Miller indices representing the in-plane lattice vector.
| example:
::

    hkl_in = [3.031127677370978, 12.31353345906843, 8.75104158816168]

- hkl_out:
| Mandatory, list with Miller indices representing the out-of-plane lattice vector.
| example:
::

    hkl_out = [9.77805918769193, -8.402719849515048, 8.436553021703112]

- twin_plane:
| Mandatory, list with Miller indices of the twin plane. If there is not a twin plane, this should be the same as sample_axis.
| example:
::

    twin_plane = [1, -1, 1]

- sample_axis:
| Mandatory, axis of the sample. The data will be rotated so that the twin plane vector in the lattice frame (hkl) corresponds to this vector in the array frame (xyz).
| example:
::

    sample_axis = [0, 1, 0]

- final_size:
| Mandatory, a size in each dimension of the array holding reconstructed object.
| example:
::

    final_size = 180


- switch_peak_trigger:
| Mandatory, a trigger defining at which iteration to switch the peak.
| example:
::

    switch_peak_trigger = [0, 50]

- mp_max_weight:
| Optional, a number between 0 and 1.0 specifying the initial coupling weight assigned to all peaks. if not provided, defaults to 1.0.
| example:
::

    mp_max_weight = 1.0

- mp_taper:
| Mandatory, fraction of the total iterations at which to begin tapering the coupling weight. Coupling weight will decrease for all peaks starting at mp_taper times the total number of iterations.
| example:
::

    mp_taper = 0.6

- lattice_size:
| Mandatory, lattice parameter of the reconstructing crystal. This is used to define the reciprocal lattice vectors, which are required for projecting to each peak.
| example:
::

    lattice_size = 0.4078

- control_peak = None
- mp_weight_init = 1.0
- mp_weight_iters = [300, 500, 700, 900, 950]
- mp_weight_vals = [0.75, 0.5, 0.3, 0.1, 0.01]
- adaptive_weights = False
- peak_threshold_init = 0.5
- peak_threshold_iters = [300, 500, 700, 900]
- peak_threshold_vals = [0.6, 0.7, 0.8, 0.9]
- rs_voxel_size = 0.0024519465219604176
- ds_voxel_size = 10.009880878864648
- calc_strain = False

