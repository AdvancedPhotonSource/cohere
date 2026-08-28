STANDARD_PREP_FIELDS = [
    {
        'key': 'data_dir', 'label': 'data dir', 'type': 'dir',
        'description': 'Directory where preprocessed data is saved.',
    },
    {
        'key': 'alien_alg', 'label': 'remove aliens algorithm', 'type': 'choice',
        'choices': ['block_aliens', 'alien_file', 'AutoAlien1'],
        'description': 'List of scans to exclude from this experiment.',
    },
    {
        'key': 'aliens', 'label': 'aliens blocks', 'type': 'list', 'element_type': 'list',
        'group': 'alien_alg.block_aliens',
        'description': 'List of blocks to be zero out, each block defined as follows: \
        [start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z].',
    },
    {
        'key': 'alien_file', 'label': 'alien file', 'type': 'file',
        'group': 'alien_alg.alien_file',
        'description': 'Mask file containing zeros for aliens.',
    },
    {
        'key': 'AA1_size_threshold', 'label': 'AA1 size threshold', 'type': 'float',
        'group': 'alien_alg.AutoAlien1',
        'description': 'Clusters of size smaller than threshold can be aliens.',
    },
    {
        'key': 'AA1_asym_threshold', 'label': 'AA1 asym threshold', 'type': 'float',
        'group': 'alien_alg.AutoAlien1',
        'description': 'Defines threshold asymetry relative to average asymetry of all clusters.',
    },
    {
        'key': 'AA1_min_pts', 'label': 'AA1 min pts', 'type': 'int',
        'group': 'alien_alg.AutoAlien1',
        'description': 'Defines minimum non zero points in neighborhood to treat the area as cluster.',
    },
    {
        'key': 'AA1_eps', 'label': 'AA1 epsilon', 'type': 'float',
        'group': 'alien_alg.AutoAlien1',
        'description': 'Defines what is neighborhood of point to look for cluster.',
    },
    {
        'key': 'AA1_amp_threshold', 'label': 'AA1 amp threshold', 'type': 'float',
        'group': 'alien_alg.AutoAlien1',
        'description': 'Amplitude threshold.',
    },
    {
        'key': 'AA1_save_arrs', 'label': 'AA1 save arrs', 'type': 'bool',
        'group': 'alien_alg.AutoAlien1',
        'description': 'Set to True only for diagnostics.',
    },
    {
        'key': 'AA1_expandcleanedsigma', 'label': 'AA1 expand cleaned sigma', 'type': 'float',
        'group': 'alien_alg.AutoAlien1',
        'description': 'If given the algorithm will apply last step of cleaning the data using the configured sigma.',
    },
    {
        'key': 'auto_intensity_threshold', 'label': 'auto intensity threshold', 'type': 'bool',
        'description': 'Set intensity threshold in auto mode.',
    },
    {
        'key': 'intensity_threshold', 'label': 'intensity threshold', 'type': 'float', 'alt_types': ['int'],
        'description': 'Intensity threshold.',
    },
    {
        'key': 'crop_pad', 'label': 'crop pad', 'type': 'list', 'element_type': 'int',
        'description': 'The data will be padded if positive, and cropped if negative \
                        as follows: [x left, x right, y left, y right, z left, z right].',
    },
    {
        'key': 'no_center_max', 'label': 'do not center max', 'type': 'bool',
        'description': 'The maximum intensity will not be centered.',
    },
    {
        'key': 'shift', 'label': 'center shift', 'type': 'list', 'element_type': 'int',
        'description': 'The data will shifted according to the given values.',
    },
    {
        'key': 'binning', 'label': 'binning', 'type': 'list', 'element_type': 'int',
        'description': 'Binning in respective axes.',
    },
]

MANDATORY = []
MANDATORY_GROUPS = ['intensity_threshold, auto_intensity_threshold']

GROUPS = {
    'alien_alg': {'block_aliens': ['aliens'], 'alien_file': ['alien_file'], 'AutoAlien1': ['AA1_size_threshold',
                    'AA1_asym_threshold', 'AA1_min_pts', 'AA1_eps', 'AA1_amp_threshold']},
}

def get_config_schema():
    return STANDARD_PREP_FIELDS