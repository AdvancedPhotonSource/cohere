MP_FIELDS = [
    {
        'key': 'scan', 'label': 'scan', 'type': 'str',
        'description': 'String type encapsulating scans or ranges of scans containing data for each peak.',
    },
    {
        'key': 'orientations', 'label': 'peaks orientations', 'type': 'list', 'element_type': 'list',
        'description': 'Each inner list defins the orientation of a peak.',
    },
    {
        'key': 'hkl_in', 'label': 'hlk in', 'type': 'list', 'element_type': 'float',
        'description': 'List with Miller indices representing the in-plane lattice vector.',
    },
    {
        'key': 'hkl_out', 'label': 'hkl out', 'type': 'list', 'element_type': 'float',
        'description': 'List with Miller indices representing the out-of-plane lattice vector.',
    },
    {
        'key': 'twin_plane', 'label': 'twin plane', 'type': 'list', 'element_type': 'int',
        'description': 'List with Miller indices of the twin plane.',
    },
    {
        'key': 'sample_axis', 'label': 'sample axis', 'type': 'list', 'element_type': 'int',
        'description': 'Axis of the sample.',
    },
    {
        'key': 'final_size', 'label': 'final size', 'type': 'int',
        'description': 'Size in each dimension of the array holding reconstructed object.',
    },
    {
        'key': 'lattice_size', 'label': 'lattice size', 'type': 'float',
        'description': 'Lattice parameter of the reconstructing crystal.',
    },
    {
        'key': 'calc_strain', 'label': 'calculate strain', 'type': 'bool',
        'description': 'Toggles whether to calculate the strain after reconstruction.',
    },
    {
        'key': 'switch_peak_trigger', 'label': 'switch peak trigger', 'type': 'trigger',
        'description': 'Defines at which iteration to apply switch peak operation.',
    },
    {
        'key': 'weight_init', 'label': 'weight init', 'type': 'float',
        'description': 'The initial global weight to use when updating the shared object',
    },
    {
        'key': 'weight_iters', 'label': 'weight iterations', 'type': 'list', 'element_type': 'int',
        'description': 'Iterations at which to update weight values.',
    },
    {
        'key': 'weight_vals', 'label': 'weight values', 'type': 'list', 'element_type': 'float',
        'description': 'Weight values corresponding to weight update iterations.',
    },
    {
        'key': 'adapt_trigger', 'label': 'adapt trigger', 'type': 'trigger',
        'description': 'Defines at which iteration to apply adapt operation.',
    },
    {
        'key': 'adapt_power', 'label': 'adapt power', 'type': 'int',
        'group': 'adapt_trigger',
        'description': 'Non-negative number that determines how harshly to punish bad datasets.',
    },
    {
        'key': 'adapt_threshold_init', 'label': 'adapt threshold init', 'type': 'float',
        'group': 'adapt_trigger',
        'description': 'Initial relative confidence threshold required for shrinkwrap to prevent bad datasets from constantly derailing the reconstruction.',
    },
    {
        'key': 'adapt_threshold_iters', 'label': 'adapt threshold iterations', 'type': 'list', 'element_type': 'int',
        'group': 'adapt_trigger',
        'description': 'Iterations to change the adapt threshold value.',
    },
    {
        'key': 'adapt_threshold_vals', 'label': 'adapt threshold values', 'type': 'list', 'element_type': 'float',
        'group': 'adapt_trigger',
        'description': 'Relative confidence threshold values corresponding to adapt_threshold_iters required for shrinkwrap to prevent bad datasets from constantly derailing the reconstruction.',
    },
    {
        'key': 'adapt_alien_start', 'label': 'adapt alien start', 'type': 'int',
        'group': 'adapt_trigger',
        'description': 'Determines when to begin adaptive alien removal.',
    },
    {
        'key': 'adapt_alien_threshold', 'label': 'adapt alien threshold', 'type': 'int',
        'group': 'adapt_trigger',
        'description': 'Determines the minimum amount of contradiction needed to mask a voxel.',
    },
]

MANDATORY = ['scan', 'orientations', 'hkl_in', 'hkl_out', 'twin_plane', 'sample_axis', 'lattice_size',
             'switch_peak_trigger', 'weight_init', ]
GROUPS = {
    'weight_iters': ['weight_vals'],
    'adapt_threshold_iters': ['adapt_threshold_vals']
}
def get_config_schema():
    return MP_FIELDS