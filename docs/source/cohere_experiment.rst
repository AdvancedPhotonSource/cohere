=================
Cohere Experiment
=================
| The experiment executed at the beamline concludes in a collection of data files and metadata for many scans. It is typical that a number of scans are selected and grouped together for analysis. The analysis of selected scans constitute of cohere experiment.
| In order to group the files associated with an cohere experiment in a structured manner user scripts will create a dedicated space. The space, cohere experiment directory, will be a sub-directory in working directory. The name of cohere experiment directory is descriptive as it contains an ID and scan ranges.
| example: with experiment ID of "ABC", and scan "56-78", the experiment directory is ABC_56-78.
| Cohere experiment can be configured for different cases. User can request to run the process separately for each scan, or scan range. User can also set the experiment to multipeak.
Refer to :ref:`main` for description of configuration parameters that set a cohere experiment.
Below we show experiment directory structure for various use cases.

Single reconstruction
+++++++++++++++++++++
|
| A directory tree that is created for a cohere experiment when using users scripts. Refer to :ref:`use` page for description and explanation how to use the scripts and to :doc:`configuration` page for configuration files and parameters.
| <experiment_dir>
|                \|
|                \|--conf
|                       \|--config
|                       \|--config_prep
|                       \|--config_data
|                       \|--config_rec
|                       \|--config_disp
|                \|--preprocessed_data
|                \|--phasing_data
|                \|--results_phasing
|                \|--results_viz
|
| The directory tree is created by users scripts.  Below is a sequence of events that produces the experiment directory tree.

- The experiment should be set up with "conf" subdirectory containing configuration files. There are scripts that create the initial experiment space.
- The script "beamline_preprocess.py" creates "prep_data.tif" file in "preprocessed_data" subdirectory. This is a file ready to be formatted.
- The script "standard_preprocess.py" reads the "preprocessed_data.tif" file, formats it, and saves the result in the "phasing_data" subdirectory as "data.tif" file. This file is ready for reconstruction.
- The "run_reconstruction.py" script reads "data.tif" file and runs phasing. The results are stored in the "results_phasing" subdirectory in "image.npy" file, along with "support.npy, and "coherence.npy" if partial coherence feature is configured.
- The "beamline_visualization.py" script loads the arrays from "results_phasing" directory, processes the image and saves it in the results_viz directory.

| After running all the scripts the experiment will have the following files:
| <experiment_dir>
|                \|
|                \|--conf
|                       \|--config
|                       \|--config_prep
|                       \|--config_data
|                       \|--config_rec
|                       \|--config_disp
|                \|--preprocessed_data
|                       \|--prep_data.tif
|                \|--phasing_data
|                       \|--data.tif
|                \|--results_phasing
|                       \|--image.npy
|                       \|--support.npy
|                       \|--image.tif
|                       \|--errors.npy
|                       \|--errors.txt
|                       \|--metrics.txt
|                \|--results_viz
|                       \|--image.vts
|                       \|--support.viz

Multiple reconstruction
+++++++++++++++++++++++
| If running multiple reconstructions which is driven by configuration (i.e. the "config_rec" file contains "reconstructions" parameter set to a number greater than 1) the "results" directory will have subdirectories reflecting the runs. The subdirectories are named by the number. The results_phasing will contain the numbered subdirectories, each with image.npy and support.npy files, and results_viz subdirectories, each with image.vts and support.vts files.
| Below is an example of "results" directory structure when running three reconstructions:
| <experiment_dir>
|                \|
|                \|--results_phasing
|                       \|--0
|                           \|--image.npy
|                           \|--support.npy
|                           \|--image.tif
|                           \|--errors.npy
|                           \|--errors.txt
|                           \|--metrics.txt
|                       \|--1
|                           \|--image.npy
|                           \|--support.npy
|                           \|--image.tif
|                           \|--errors.npy
|                           \|--errors.txt
|                           \|--metrics.txt
|                       \|--2
|                           \|--image.npy
|                           \|--support.npy
|                           \|--image.tif
|                           \|--errors.npy
|                           \|--errors.txt
|                           \|--metrics.txt
|                \|--results_viz
|                       \|--0
|                           \|--image.vts
|                           \|--support.vts
|                       \|--1
|                           \|--image.vts
|                           \|--support.vts
|                       \|--2
|                           \|--image.vts
|                           \|--support.vts

Genetic Algorithm
+++++++++++++++++
| When running GA, only the best results are saved. Phasing results are saved in results_phasing directory, and visualization results are saved in results_viz.

Separate scans
++++++++++++++
| When the experiment is configured as separate reconstruction for each scan, the experiment directory will contain a subdirectory for each scan. This use case is configured in "config_prep" file by setting parameter "separate_scans" to True. Each scan directory is a concatination of "scan", underscore, and the scan number. Each of the scan subdirectories will have preprocessed_data, phasing_data, results_phasing, and results_viz subdirectories. The configuration is common for all scans. If running multiple reconstructions or GA, the directory structure in each scan directory will reflect it, as described in above sections.
| Below is an example of directory structure for separate scans.
| <experiment_dir>
|                \|
|                \|--conf
|                       \|--config
|                       \|--config_prep
|                       \|--config_data
|                       \|--config_rec
|                       \|--config_disp
|                \|--scan_54
|                       \|--preprocessed_data
|                             \|--prep_data.tif
|                       \|--phasing_data
|                             \|--data.tif
|                       \|--results_phasing
|                             \|--image.npy
|                             \|--support.npy
|                             \|--image.tif
|                             \|--errors.npy
|                             \|--errors.txt
|                             \|--metrics.txt
|                        \|--results_viz
|                             \|--image.vts
|                             \|--support.vts
|                \|--scan_57
|                       \|--preprocessed_data
|                             \|--prep_data.tif
|                       \|--phasing_data
|                             \|--data.tif
|                       \|--results_phasing
|                             \|--image.npy
|                             \|--support.npy
|                             \|--image.tif
|                             \|--errors.npy
|                             \|--errors.txt
|                             \|--metrics.txt
|                       \|--results_viz
|                             \|--image.vts
|                             \|--support.vts

Alternate configuration
+++++++++++++++++++++++
| The "run_rec.py" script supports feature of running reconstruction with alternate configuration(s). Each alternate configuration must be named with arbitrary postfix (rec_id), preceded by "confic_rec" and underscore. This file should be created in the conf subdirectory. Refer to 'Scripts'  section below for instruction how to run a case with alternate reconstruction configuration.
| After running the "run_rec" script with this option, the results will be saved in the results_phasing_<rec_id> directory.
| Below is an example of directory structure with alternate configuration.
| <experiment_dir>
|                \|
|                \|--conf
|                       \|--config
|                       \|--config_prep
|                       \|--config_data
|                       \|--config_rec
|                       \|--config_rec_aa
|                       \|--config_rec_bb
|                       \|--config_disp
|                \|--prepprocessed_data
|                       \|--prep_data.tif
|                \|--phasing_data
|                       \|--data.tif
|                \|--results_phasing
|                       \|--image.npy
|                       \|--support.npy
|                       \|--image.tif
|                       \|--errors.npy
|                       \|--errors.txt
|                       \|--metrics.txt
|                \|--results_viz
|                       \|--image.vts
|                       \|--support.viz
|                \|--results_phasing_aa
|                       \|--image.npy
|                       \|--support.npy
|                       \|--image.tif
|                       \|--errors.npy
|                       \|--errors.txt
|                       \|--metrics.txt
|                \|--results_viz_aa
|                       \|--image.vts
|                       \|--support.viz
|                \|--results_phasing_bb
|                       \|--image.npy
|                       \|--support.npy
|                       \|--image.tif
|                       \|--errors.npy
|                       \|--errors.txt
|                       \|--metrics.txt
|                \|--results_viz_bb
|                       \|--image.vts
|                       \|--support.viz

