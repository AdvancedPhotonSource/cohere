RECON_FIELDS = {
    'general': [
    {
        'key': 'data_dir', 'label': 'data dir', 'type': 'dir',
        'description': 'Defines directory from which data is read.',
    },
    {
        'key': 'save_dir', 'label': 'save dir', 'type': 'dir',
        'description': 'Defines directory where results of reconstruction are saved.',
    },
    {
        'key': 'init_guess', 'label': 'initial guess', 'type': 'choice',
        'choices': ['random', 'continue', 'AI_guess'],
        'description': 'Defines how to set the initial image.',
    },
    {
        'key': 'continue_dir', 'label': 'continue dir', 'type': 'dir',
        'group': 'init_guess.continue',
        'description': 'Defines directory from which results are read for reconstruction continuation.',
    },
    {
        'key': 'AI_trained_model', 'label': 'AI trained model file', 'type': 'file',
        'group': 'init_guess.AI_guess',
        'description': 'Defines the h5 file that holds trained model.',
    },
    {
        'key': 'reconstructions', 'label': 'reconstruction number', 'type': 'int',
        'description': 'Number of reconstructions.',
    },
    {
        'key': 'processing', 'label': 'processing backend', 'type': 'choice',
        'choices': ['auto', 'cp', 'torch', 'np'],
        'description': 'Library used when running reconstruction.',
    },
    {
        'key': 'device', 'label': 'GPU IDs', 'type': 'list', 'element_type': 'int',
        'alt_types': ['str', 'dict'],
        'description': 'Defines GPU id(s) that should be used for reconstruction(s) when using cupy or torch. \
                       Supported formats: [1,2,3], "all", {"host1":"all", "host2":[0,1,2,3,4]}',
    },
    {
        'key': 'algorithm_sequence', 'label': 'algorithm sequence', 'type': 'str',
        'description': 'String defining iteration sequence.',
    },
    {
        'key': 'hio_beta', 'label': 'hio beta', 'type': 'float',
        'description': 'Parameter used in HIO algorithm.',
    },
    {
        'key': 'raar_beta', 'label': 'raar beta', 'type': 'float',
        'description': 'Parameter used in RAAR algorithm.',
    },
    {
        'key': 'initial_support_area', 'label': 'initial support area', 'type': 'list', 'element_type': 'float',
        'description': 'Defines setting of initial support area as fraction of the array by axis.',
    },
    ],

    'features': [
    {
        'key': 'ga_generations', 'label': 'number of generations', 'type': 'int',
        'description': 'Number of generations',
    },
    {
        'key': 'ga_metrics', 'label': 'GA metrics', 'type': 'list', 'element_type': 'choice',
        'choices': ['chi', 'area', 'summed_phase', 'sharpness'],
        'group': 'ga_generations',
        'description': 'Defines which metric should be used to rank the reconstruction results.',
    },
    {
        'key': 'ga_breed_modes', 'label': 'breeding algorithm', 'type': 'list', 'element_type': 'choice',
        'choices': ['sqrt_ab', 'pixel_switch', 'b_pa', '2ab_a_b', '2a_b_pa', 'sqrt_ab_pa', 'sqrt_ab_recip', \
                    'max_ab', 'max_ab_pa', 'avg_ab', 'avg_ab_pa'],
        'group': 'ga_generations',
        'description': 'Defines which breeding mode to use to populate new generation.',
    },
    {
        'key': 'ga_cullings', 'label': 'cullings', 'type': 'list', 'element_type': 'int',
        'group': 'ga_generations',
        'description': 'Defines how many worst samples to remove in breeding phase for each generation.',
    },
    {
        'key': 'ga_sw_thresholds', 'label': 'shrink wrap threshold', 'type': 'list', 'element_type': 'float',
        'group': 'ga_generations',
        'description': 'The threshold to apply when recalculating support after breeding for each generation.',
    },
    {
        'key': 'ga_sw_gauss_sigmas', 'label': 'shrink wrap sigma', 'type': 'list', 'element_type': 'float',
        'group': 'ga_generations',
        'description': 'The Gauss sigma to apply when recalculating support after breeding for each generation.',
    },
    {
        'key': 'ga_lpf_sigmas', 'label': 'lowpass filter sigma', 'type': 'list', 'element_type': 'float',
        'group': 'ga_generations',
        'description': 'The Gauss sigma of lowpass filter for each generation.',
    },
    {
        'key': 'ga_gen_pc_start', 'label': 'generation to start pc', 'type': 'int',
        'group': 'ga_generations',
        'description': 'Defines generation at which pcdi feature will start, if active.',
    },
    {
        'key': 'ga_fast', 'label': 'True for fast GA, False for populous', 'type': 'bool',
        'group': 'ga_generations',
        'description': 'Defines which GA algorithm to use.',
    },

    {
        'key': 'twin_trigger', 'label': 'twin trigger', 'type': 'trigger',
        'description': 'Defines at which iteration to apply twin operation.',
    },
    {
        'key': 'twin_halves', 'label': 'twin halves', 'type': 'list', 'element_type': 'int',
        'group': 'twin_trigger',
        'description': 'Defines which half of the array is zeroed out in each dimension.',
    },

    {
        'key': 'shrink_wrap_trigger', 'label': 'shrink wrap trigger', 'type':'trigger',
        'description': 'Defines at which iteration to apply shrink wrap operation.',
    },
    {
        'key': 'shrink_wrap_type', 'label': 'shrink wrap type', 'type': 'str', 'alt_types': ['list'],
        'group': 'shrink_wrap_trigger',
        'description': 'Type of shrink wrap, currently supporting GAUSS.',
    },
    {
        'key': 'shrink_wrap_threshold', 'label': 'shrink wrap threshold', 'type': 'float', 'alt_types': ['list'],
        'group': 'shrink_wrap_trigger',
        'description': 'Threshold for Gauss filter.',
    },
    {
        'key': 'shrink_wrap_gauss_sigma', 'label': 'shrink wrap sigma', 'type': 'float', 'alt_types': ['list'],
        'group': 'shrink_wrap_trigger',
        'description': 'Sigma for Gauss filter.',
    },

    {
        'key': 'phc_trigger', 'label': 'phase constrain trigger', 'type':'trigger',
        'description': 'Defines at which iteration to apply phase constrain operation.',
    },
    {
        'key': 'phc_phase_min', 'label': 'min phase', 'type': 'float', 'alt_types': ['list'],
        'group': 'phc_trigger',
        'description': 'Defines lower threshold.',
    },
    {
        'key': 'phc_phase_max', 'label': 'max phase', 'type': 'float', 'alt_types': ['list'],
        'group': 'phc_trigger',
        'description': 'Defines upper threshold.',
    },

    {
        'key': 'pc_interval', 'label': 'partial coherence interval', 'type':'int',
        'description': 'Defines iteration interval to update coherence.',
    },
    {
        'key': 'pc_type', 'label': 'partial coherence type', 'type': 'str',
        'group': 'phc_trigger',
        'description': 'Partial coherence algorithm. Currently LUCY type is supported.',
    },
    {
        'key': 'pc_LUCY_iterations', 'label': 'LUCY iteration number', 'type': 'int',
        'group': 'phc_trigger',
        'description': 'Defines number of iteration when running LUCY algorithm.',
    },
    {
        'key': 'pc_normalize', 'label': 'partial coherence normalize', 'type': 'bool',
        'group': 'phc_trigger',
        'description': 'Normalize result of partial coherence. Typically set to True',
    },
    {
        'key': 'pc_LUCY_kernel', 'label': 'LUCY kernel', 'type': 'list', 'element_type': 'int',
        'group': 'phc_trigger',
        'description': 'Coherence kernel area.',
    },

    {
        'key': 'lowpass_filter_trigger', 'label': 'lowpass filter trigger', 'type':'trigger',
        'description': 'Defines at which iteration to apply lowpass filter operation.',
    },
    {
        'key': 'lowpass_filter_range', 'label': 'lowpass filter range', 'type': 'list', 'element_type': 'float',
        'group': 'lowpass_filter_trigger',
        'description': 'Defines range of lowpass filter sigma.',
    },

    {
        'key': 'average_trigger', 'label': 'average trigger', 'type':'trigger',
        'description': 'Defines at which iteration to apply average operation.',
    },

    {
        'key': 'progress_trigger', 'label': 'progress trigger', 'type':'trigger',
        'description': 'Defines at which iteration to apply progress operation.',
    },

    {
        'key': 'live_trigger', 'label': 'live trigger', 'type':'trigger',
        'description': 'Defines at which iteration to apply live operation.',
    },
    ]
}

MANDATORY = ['algorithm_sequence']
GROUPS = {
    'init_guess': {'random': [], 'continue': ['continue_dir'], 'AI_guess': ['AI_trained_model']},
    'ga_generations': ['ga_metrics', 'ga_breed_modes', 'ga_sw_thresholds', 'ga_sw_gauss_sigmas'],
    'twin_trigger': ['twin_halves'],
    'shrink_wrap_trigger': ['shrink_wrap_type', 'shrink_wrap_threshold', 'shrink_wrap_gauss_sigma'],
    'phc_trigger': ['phc_phase_min', 'phc_phase_max'],
    'pc_interval': ['pc_type', 'pc_LUCY_iterations', 'pc_normalize', 'pc_LUCY_kernel'],
    'lowpass_filter_trigger': ['lowpass_filter_range'],
}

def get_config_schema():
    return RECON_FIELDS