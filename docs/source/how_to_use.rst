.. _use:

Use
===
| The user scripts can be obtained from https://github.com/AdvancedPhotonSource/cohere-scripts. The cohere-scripts package, is a supplemental git repository that contains executable python scripts for every stage of CDI phase retrieval, from raw data processing to final images. Part of package are classes encapsulating instruments that are used to collect experiment data at the 34-ID-C beamline. User can easily add own code that would encapsulate user's beamline specifics. 
In addition the repository contains directory with sample configuration files for each stage, and a complete example including experimental data, and configuration files to phase the data.

Installing Scripts
##################
| There are three ways to get the cohere-scripts package onto your local computer, or remote computer.

1. The cohere-scripts package can be downloaded as a zip file by visiting https://github.com/advancedPhotonSource/cohere-scripts and clicking the green “Code” button and selecting “Download ZIP” as shown below.

2. Alternatively one can clone the repository using git. This will create the cohere-scripts directory containing all of the cohere-scripts content. In this code below we clone it to cohere-scripts-main directory.
   
  ::

        git clone https://github.com/advancedPhotonSource/cohere-scripts cohere-scripts-main

3. Another way to get the cohere-scripts package on your machine is with the wget command:
   
  ::

        wget https://github.com/AdvancedPhotonSource/cohere-scripts/archive/main.zip
        unzip main.zip

| After the package is installed change directory to cohere-scripts-main and run setup.py script. The setup.py script modifies paths from relative to absolute in the provided example configuration.:
   
  ::

        cd cohere-scripts-main
        python setup.py

Scripts
####### 
| In this documentation it is assumed the scripts are invoked from cohere-scripts directory. Below is a list of scripts with description and explanation how to run:

- create_experiment.py

  This script creates a new experiment directory with conf subdirectory. The conf contains all configuration files. The files contain all parameters, but most of them commented out to make them inactive. User needs to edit the configuration files and un-comment to use wanted functionality.
  Running this script:
  ::

        python scripts/create_experiment.py <id> <scan range> <working_dir> --specfile <specfile> --beamline <beamline>

  The parameters are as follows:
     * id: an arbitrary literal value assign to this experiment
     * scan range: scans that will be included in the data. This can be a single scan or range separated with "-"
     * working_dir: a directory where the experiment will be created
     * specfile: optional, but typically used for beamline aps_34idc
     * beamline: optional, specifies at which beamline the experiment was performed. If not configured, the preparation and visualization is not available.

- setup_34idc.py

  This script creates a new experiment directory structure.
  Running this script:
  ::

        python scripts/setup_34idc.py <id> <scan range> <conf_dir> --specfile <specfile> --copy_prep

  The parameters are as follows:
     * id: an arbitrary literal value assign to this experiment
     * scan range: scans that will be included in the data. This can be a single scan or range separated with "-"
     * conf_dir: a directory from which the configuration files will be copied
     * specfile: optional, used when specfile configured in <conf_dir>/config file should be replaced by another specfile
     * copy_prep: this is a switch parameter, set to true if the prep_data.tif file should be copied from experiment with the <conf_dir> into the prep directory of the newly created experiment

- beamline_preprocess.py

  To run this script a configuration file "config_prep" must be defined in the <experiment_dir>/conf directory and the main configuration "config" file must have beamline parameter configured. This script reads raw data, applies correction based on physical properties of the instrument, and optionally aligns and combines multiple scans. The prepared data file is stored in <experiment_dir>/prep/prep_data.tif file.
  note: when separate_scan is configured to true, a prep_data.tiff file is created for each scan.
  Running this script:
  ::

        python scripts/beamline_preprocess.py <experiment_dir>

  The parameters are as follows:
     - experiment directory: directory of the experiment space

- standard_preprocess.py

  To run this script a configuration file "config_data" must be defined in the <experiment_dir>/conf directory, and the "prep_data.tif" file must be present in experiment space. This script reads the prepared data, formats the data according to configured parameters, and produces data.tif file. The file is stored in <experiment_dir>/data/data.tif file.
  Running this script:
  ::

        python scripts/standard_preprocess.py <experiment_dir>

  The parameters are as follows:
     * experiment directory: directory of the experiment space

- run_reconstruction.py

  To run this script a configuration file "config_rec" must be defined in the <experiment_dir>/conf directory, and the "data.tif" file must be present in experiment space. This script reads the data file and runs the reconstruction software. The reconstruction results are saved in <experiment_dir>/results directory.
  note: The results might be saved in different location in experiment space, depending on the use case. Refer to 'Experiment' section for details.
  Running this script:
  ::

        python scripts/run_reconstruction.py <processor> <experiment_dir> --rec_id <alternate reconstruction id>

  The parameters are as follows:
     * processor: the library used when running reconstruction. Possible options:

       + auto
       * np
       * cp
       * af
       * cuda
       + opencl
       + cpu

       When the auto option is selected the program will use the fastest library that is available, in this order: cupy, af, numpy. The af option leaves the choice to arrayfire, which will select a library in this order: cuda, opencl, cpu. The cp option will utilize cupy, np will utilize numpy, and af will leave selection to arrayfire. The cuda, opencl, and cpu are arrayfire libraries. The "cuda" and "opencl" options will invoke the processing on GPUs, and the "cpu" option on cpu.
     * experiment directory: directory of the experiment space
     * rec_id: optional parameter, when present, the alternate configuration will be used to run reconstruction

- beamline_visualization.py

  To run this script a configuration file "config_disp" must be defined in the <experiment_dir>/conf directory, the main configuration "config" file must have beamline parameter configured, and the reconstruction must be completed. This script reads the reconstructed files, and processes them to create .vts files that can be viewed utilizing visualization tools such Paraview. The script will process "image.npy" files that are in the experiment space and in a subdirectory of "resuls_dir" configuration parameter, or a given file is --image_file option is used.
  Running this script:
  ::

        python scripts/beamline_visualization.py <experiment_dir> --image_file <image_file>

  The parameters are as follows:
     * experiment directory: directory of the experiment space
     * image_file: optional parameter, if given, this file will be processed.

- everything.py

  To run this script all configuration files must be defined. This script runs the scripts in the following order: run_prep.py, format_data.py, run_rec.py, and run_disp.py. If the beamline parameter is not defined in the experiment main configuration file "config", the run_prep.py and run_disp.py scripts will be omitted, as they are customized for a beamline. The experiment space must be already defined.
  Running this script:
  ::

        python scripts/everything.py <processor> <experiment_dir> --rec_id <alternate reconstruction id>

  The parameters are as follows:
     * experiment directory: directory of the experiment space
     * processor: the library used when running reconstruction.
     * rec_id: optional parameter, when present, the alternate configuration will be used to run reconstruction

- cdi_window.py

  This script starts GUI that offers complete interface to run all the scripts described above. In addition GUI interface offers easy way to modify configuration.
  Running this script:
  ::

        python scripts/cdi_window.py

