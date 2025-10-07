=========
config_mp
=========

**Data/geometry parameters**

- scan

| Mandatory, string type encapsulating scans or ranges of scans containing data for each peak. The scans/scan ranges should be arranged in ascending order.

::

    scan = "898-913,919-934,940-955,961-976"

- orientations

| Mandatory, a list of lists, each inner list defining the orientation of a peak.

::

    orientations = [[-1, -1, 1], [0, 0, 2], [1, 1, 1], [2, 0, 0]]

- hkl_in

| Mandatory, list with Miller indices representing the in-plane lattice vector.

::

    hkl_in = [3.031127677370978, 12.31353345906843, 8.75104158816168]

- hkl_out

| Mandatory, list with Miller indices representing the out-of-plane lattice vector.

::

    hkl_out = [9.77805918769193, -8.402719849515048, 8.436553021703112]

- twin_plane

| Mandatory, list with Miller indices of the twin plane. If there is not a twin plane, this should be the same as sample_axis.

::

    twin_plane = [1, -1, 1]

- sample_axis

| Mandatory, axis of the sample. The data will be rotated so that the twin plane vector in the lattice frame (hkl) corresponds to this vector in the array frame (xyz).

::

    sample_axis = [0, 1, 0]

- final_size

| Mandatory, a size in each dimension of the array holding reconstructed object.

::

    final_size = 180

- lattice_size

| Mandatory, lattice parameter of the reconstructing crystal. This is used to define the reciprocal lattice vectors, which are required for projecting to each peak.

::

    lattice_size = 0.4078

- rs_voxel_size
- ds_voxel_size

| Autofilled, These are calculated from the experimental geometry and added to the config file automatically. Do not edit these unless you REALLY know what you're doing.

**General reconstruction parameters**

- switch_peak_trigger

| Mandatory, a trigger defining at which iteration to switch the peak.

::

    switch_peak_trigger = [0, 50]

- weight_init

| Mandatory, the initial global weight to use when updating the shared object. A weight of 0 will not update the shared object at all, while a weight of 1 will replace the shared object (except the orthogonal displacement) after phasing each peak. In general, a high weight leads to rapid development, but only forces the object to agree with whichever peak it phased most recently. A low weight converges slowly, but favors information common to all of the peaks.

::
    weight_init = 1.0

- weight_iters
- weight_vals

| Optional, list of iterations and corresponding list of values. After each iteration in weight_iters, the global weights will be updated to the corresponding value in weight_vals. A good rule of thumb is to start high and end low.

::
    weight_iters = [200, 400, 600, 800]
    weight_vals = [0.75, 0.5, 0.25, 0.1]

- calc_strain

| Optional, boolean, toggles whether to calculate the strain after reconstruction.

::

    calc_strain = False

**Adaptive reconstruction parameters**

- adapt_trigger

| Optional, determines when to update the weights assigned to individual peaks. Before each peak-switch, a confidence value is calculated for the current peak, based on how much it changed while phasing (i.e. how much this dataset disagrees with the current state of the reconstruction). These are then periodically compiled into a peak-specific weight, which determines how much each dataset is allowed to impact the reconstruction.

::

    adapt_trigger = [100, 100]

- adapt_power

| Optional, non-negative number that determines how harshly to punish bad datasets.

::

    adapt_power = 2  #

- adapt_threshold_init
- adapt_threshold_iters
- adapt_threshold_vals

| Optional, determines the relative confidence threshold required for shrinkwrap to prevent bad datasets from constantly derailing the reconstruction.

::

    adapt_threshold_init = 0.5
    adapt_threshold_iters = [200, 400, 600, 800]
    adapt_threshold_vals = [0.6, 0.7, 0.8, 0.9]

- adapt_alien_start

| Optional, determines when to begin adaptive alien removal. Alien removal occurs immediately after switching to a new peak, while the exit wave still agrees perfectly with the shared object. The diffraction amplitude calculated by forward propagating this object is compared with the actual measurement of the same reflection, and voxels where they strongly contradict each other are masked. A hybrid diffraction pattern, where the masked voxels have been replaced by the forward propagation, is then used for phasing.

::

    adapt_alien_start = 50

- adapt_alien_threshold

| Optional, determines the minimum amount of contradiction needed to mask a voxel. Under the hood, the actual value is this multiplied by the median of the normalized difference map.

::

    adapt_alien_threshold = 2
