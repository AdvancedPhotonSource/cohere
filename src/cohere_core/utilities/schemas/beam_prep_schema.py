BEAM_PREP_FIELDS = [
        {
            'key': 'min_frames', 'label': 'minimum frames', 'type': 'int',
            'description': 'Only scans containing minimum of defined number of frames will be included.',
        },
        {
            'key': 'exclude_scans', 'label': 'exclude scans', 'type': 'list', 'element_type': 'int',
            'description': 'list of scans to exclude from this experiment.',
        },
        {
            'key': 'remove_outliers', 'label': 'remove outliers', 'type': 'bool',
            'description': 'Dictates whether auto discovery and exclusion of data with low correlation should be run.',
        },
        {
            'key': 'outliers_scans', 'label': 'outliers_scans', 'type': 'list', 'element_type': 'int', 'set': 'False',
            'description': 'list of scans found to be outliers.',
        },
        {
            'key': 'roi', 'label': 'user defined roi', 'type': 'list', 'element_type': 'int',
            'group': 'roi_format',
            'description': 'ROI info conforming to the ROI format.',
        },
        {
            'key': 'roi_format', 'label': 'roi format', 'type': 'str',
            'description': 'Interprets the roi entry. Supported values: \
                           "center_point_dist" : [center_point_x, center_point_y, distance_x, distance_y], \
                           "start_point_end_point" : [start_point_x, start_point_y, end_point_x, end_point_y], \
                           "start_point_distance" : [start_point_x, distance_x, start_point_y, distance_y].',
        },
        {
            'key': 'max_crop', 'label': 'max crop', 'type': 'list', 'element_type': 'int',
            'description': 'Defines size of frame cut out around maximum.',
        },
        {
            'key': 'do_RSM', 'label': 'do RSM', 'type': 'bool',
            'description': 'Defines if the reciprocal space mapping will be generated.',
        },
    ]

MANDATORY = [] # There are mandatory parameters but they are checked in the code

def get_config_schema():
    return BEAM_PREP_FIELDS