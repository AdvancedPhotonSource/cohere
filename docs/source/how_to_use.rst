.. _use:

Using cohere with cohere-ui package
========================================
| User scripts can be obtained from https://github.com/AdvancedPhotonSource/cohere-ui. The cohere-ui package, is a supplemental git repository that contains executable python scripts for every stage of BCDI phase retrieval, from raw data processing to final images. Part of package are classes encapsulating instruments that are used to collect experiment data at the 34-ID-C beamline. User with other instruments than those defined in 34-ID-C suite can easily add own code that would encapsulate user's beamline specifics. In addition the repository contains directory with sample configuration files for each stage, and a complete example including experimental data, and configuration files to phase the data.

Installing Scripts
##################
| There are three ways to get the cohere-ui package onto your local computer, or remote computer.
| User should activate the environment where cohere was installed.

  ::

    conda activate <env_name>

1. The cohere-ui package can be downloaded as a zip file by visiting https://github.com/advancedPhotonSource/cohere-ui and clicking the green “Code” button and selecting “Download ZIP”. Unzip this directory and rename to cohere-ui or a name of your choice.

2. Alternatively one can clone the repository using git. This will create the cohere-ui directory containing all of the cohere-ui content. In this code below we clone it to cohere-ui directory.

  ::

        git clone https://github.com/advancedPhotonSource/cohere-ui

3. Another way to get the cohere-ui package on your machine is with the wget command. After unzipping rename the directory to cohere-ui or a name of your choice.

  ::

        wget https://github.com/AdvancedPhotonSource/cohere-ui/archive/main.zip
        unzip main.zip

| After the package is installed run setup.py and install_pkgs.sh/install_pkgs.bat scripts. The setup.py script modifies paths from relative to absolute in the provided example configuration. The install_pkgs script installs python packages xrayutilities, mayavi, and pyqt that are required to run the cohere-ui.

  ::

        cd cohere-ui
        python setup.py
        sh install_pkgs.sh    # for Linux and OS_X
        install_pkgs.bat      # for Windows

Scripts
####### 
| In this documentation it is assumed the scripts are invoked from cohere-ui directory. Below is a list of scripts with description and explanation how to run them:

- create_experiment.py

  This script creates a new experiment directory with conf subdirectory. The conf contains all configuration files. The files contain all parameters, but most of them commented out to make them inactive. User needs to edit the configuration files and un-comment to use wanted functionality.
  Running this script:
  ::

        python cohere-scripts/create_experiment.py <id> <scan ranges> <working_dir> --specfile <specfile> --beamline <beamline>

  The parameters are as follows:
     * id: an arbitrary literal value assign to this experiment
     * scan ranges: scans that will be included in the data. This can be a single number, a range separated with "-", or combination of numbers and ranges separated by comma.
     * working_dir: a directory where the experiment will be created
     * specfile: optional, but typically used for beamline aps_34idc
     * beamline: optional, specifies at which beamline the experiment was performed. If not configured, the beamline preprocessing and visualization is not available.

- setup_34idc.py

  This script creates a new experiment directory structure.
  Running this script:
  ::

        python cohere-scripts/setup_34idc.py <id> <scan range> <conf_dir> --specfile <specfile> --copy_prep

  The parameters are as follows:
     * id: an arbitrary literal value assign to this experiment
     * scan range: scans that will be included in the data. This can be a single number, a range separated with "-", or combination of numbers and ranges separated by comma.
     * conf_dir: a directory from which the configuration files will be copied
     * specfile: optional, used when specfile configured in <conf_dir>/config file should be replaced by another specfile
     * copy_prep: this is a switch parameter, set to True if the prep_data.tif file should be copied from experiment with the <conf_dir> into the prep directory of the newly created experiment

- beamline_preprocess.py

  To run this script a configuration file "config_prep" must be defined in the <experiment_dir>/conf directory and the main configuration "config" file must have beamline parameter configured. This script reads raw data, applies correction based on physical properties of the instrument used during experiment, and optionally aligns and combines multiple scans. The prepared data file is stored in <experiment_dir>/preprocessed_data/prep_data.tif file.
  note: when separate_scan is configured to True, a prep_data.tiff file is created for each scan.
  Running this script:
  ::

        python cohere-scripts/beamline_preprocess.py <experiment_dir>

  The parameters are as follows:
     - experiment directory: directory of the experiment space

- standard_preprocess.py

  To run this script a configuration file "config_data" must be defined in the <experiment_dir>/conf directory, and the "prep_data.tif" file must be present in experiment space. This script reads the preprocessed data, formats the data according to configured parameters, and produces data.tif file. The file is stored in <experiment_dir>/phasing_data/data.tif file.
  Running this script:
  ::

        python cohere-scripts/standard_preprocess.py <experiment_dir>

  The parameters are as follows:
     * experiment directory: directory of the experiment space

- run_reconstruction.py

  To run this script a configuration file "config_rec" must be defined in the <experiment_dir>/conf directory, and the "data.tif" file must be present in experiment space. This script reads the data file and executs phasing script. The reconstruction results are saved in <experiment_dir>/results_phasing directory.
  note: The results might be saved in different location in experiment space, depending on the use case. Refer to 'Experiment' section for details.
  Running this script:
  ::

        python cohere-scripts/run_reconstruction.py <experiment_dir> --rec_id <alternate reconstruction id>

  The parameters are as follows:
     * experiment directory: directory of the experiment space.
     * rec_id: optional parameter, when present, the alternate configuration will be used to run reconstruction. . Refer to 'Experiment' section for details.

- beamline_visualization.py

  To run this script a configuration file "config_disp" must be defined in the <experiment_dir>/conf directory, the main configuration "config" file must have beamline parameter configured, and the reconstruction must be completed. This script reads the reconstructed files, and processes them to create .vts files that can be viewed utilizing visualization tools such Paraview. The script will process "image.npy" files that are in the experiment space that is defined by the <experiment_dir>. If "resuls_dir" configuration parameter is defined in config_disp, then the program will find and process all image.npy files in that directory tree, otherwise it will find and process all image.npy files in experiment directory tree. If rec_id parameter is present, the script will find and process all image.npy files in directory tree startin with <experiment_dir>/results_pasing_<rec_id>. If --image_file option is used the programm will process the given single file.
  Running this script:
  ::

        python cohere-scripts/beamline_visualization.py <experiment_dir> --rec_id <reconstruction id> --image_file <image_file>

  The parameters are as follows:
     * experiment directory: directory of the experiment space
     * rec_id: optional, id of alternate reconstruction, defined by alternate configuration file rec_config_<rec_id>
     * image_file: optional parameter, if given, this file will be processed.

- everything.py

  To run this script all configuration files must be defined. This script runs the scripts in the following order: beamline_preprocess.py, standard_preprocess.py, run_reconstruction.py, and beamline_visualization.py. If the beamline parameter is not defined in the experiment main configuration file "config", the beamline_preprocess.py and beamline_visualization.py scripts will be omitted, as they are customized for a beamline.
  Running this script:
  ::

        python cohere-scripts/everything.py <experiment_dir> --rec_id <reconstruction id>

  The parameters are as follows:
     * experiment directory: directory of the experiment space
     * rec_id: optional parameter, when present, the alternate configuration will be used to run reconstruction

- cdi_window.py

  This script starts GUI that offers complete interface to run all the scripts described above. In addition GUI interface offers easy way to modify configuration.
  Running this script:
  ::

        python cohere-scripts/cdi_window.py

