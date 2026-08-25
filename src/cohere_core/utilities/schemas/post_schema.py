POST_FIELDS = {
    'general': [
    {
        'key': 'results_dir', 'label': 'data dir', 'type': 'dir',
        'description': 'Results in this directory tree will be used as input to the postprocessing.',
    },
    {
        'key': 'rampups', 'label': 'rampups', 'type': 'int',
        'description': 'Upsize parameter when running ramp removal.',
    },
    {
        'key': 'unwrap', 'label': 'unwrap', 'type': 'bool',
        'description': 'If True the image.vts file will contain unwrapped phase.',
    },
    {
        'key': 'make_twin', 'label': 'make twin', 'type': 'bool',
        'description': 'If True another output is generated for twin image.',
    },
        {
            'key': 'complex_mode', 'label': 'complex mode', 'type': 'choice', 'choices': ['AmpPhase', 'ReIm'],
            'description': 'The mode determines arrays that will be saved in the direct space images file. \
                       If not given, it will default to AmpPhase.',
        },
    ],

    'features': [
        {
            'key': 'crop_type', 'label': 'crop type', 'type': 'choice', 'choices': ['fraction', 'tight'],
            'description': 'Defines how cropshould be set.',
        },
        {
            'key': 'crop_margin', 'label': 'crop margin', 'type': 'int',
            'group': 'crop_type.tight',
            'description': 'The margin will be added to each side of the extend array.',
        },
        {
            'key': 'crop_thresh', 'label': 'crop thresh', 'type': 'float',
            'group': 'crop_type.tight',
            'description': 'The threshold used when determining the extend array.',
        },
        {
            'key': 'crop_fraction', 'label': 'crop fraction', 'type': 'list', 'element_type': 'float',
            'group': 'crop_type.fraction',
            'description': 'Defines size of the cropped array relative to the full image array. \
                            The full array is cropped around maximum value.',
        },
        {
            'key': 'interpolation_mode', 'label': 'interpolation_mode', 'type': 'choice', 'choices': ['AmpPhase', 'ReIm'],
            'description': 'If defined, the cohere will proceed with image interpolation according to defined mode.',
        },
        {
            'key': 'interpolation_resolution', 'label': 'interpolation resolution', 'type': 'str',
            'alt_types': ['int', 'float', 'list'],
            'group': 'interpolation_mode',
            'description': 'Required parameter for interpolation.',
        },
        {
            'key': 'determine_resolution_type', 'label': 'determine resolution type', 'type': 'choice',
            'choices': ['deconv'],
            'description': '',
        },
        {
            'key': 'resolution_deconv_contrast', 'label': 'resolution deconv contrast', 'type': 'float',
            'group': 'determine_resolution_type',
            'description': '',
        },
        {
            'key': 'write_recip', 'label': 'write recip', 'type': 'bool',
            'description': 'If True the reciprocal_space.vts file will be saved with arrays of \
                           phasing data and inverse fourier of that data.',
        },
        {
            'key': 'Bragg_displacement', 'label': 'Bragg displacement', 'type': 'str',
            'alt_types': ['float'],
            'description': 'If configured the vts file with image will also include displacement array.',
        },
        {
            'key': 'compute_strain', 'label': 'compute strain', 'type': 'bool',
            'description': 'If True the vts file with image will also include strain.',
        },
    ]
}

GROUPS = {
    'crop_type': {'tight': ['crop_margin', 'crop_thresh'], 'fraction': ['crop_fraction']},
    'interpolation_mode': ['interpolation_resolution'],
    'determine_resolution_type': ['resolution_deconv_contrast']
}

def get_config_schema():
    return POST_FIELDS