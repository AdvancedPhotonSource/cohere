EXP_FIELDS = [
        {
            'key': 'working_dir', 'label': 'working directory', 'type': 'dir',
            'description': 'Directory containing experiment subdirectory.',
        },
        {
            'key': 'experiment_id', 'label': 'experiment ID', 'type': 'str',
            'description': 'Arbitrary string assigned to this experiment.',
        },
        {
            'key': 'scan', 'label': 'experiment scans', 'type': 'str',
            'description': 'A single number, a range, or combination of numbers and /'
                           'ranges separated by comma, defining scans included in this cohere experiment.',
        },
        {
            'key': 'beamline', 'label': 'beamline', 'type': 'str',
            'description': 'Beamline that the experiment was conducted on.',
        },
        {
            'key': 'separate_scans', 'label': 'separate scans', 'type': 'bool',
            'description': 'Defines if the scans in cohere experiment should be processed separately.',
        },
        {
            'key': 'separate_scan_ranges', 'label': 'separate scan ranges', 'type': 'bool',
            'description': 'Defines if the scan ranges in cohere experiment should be processed separately.',
        },
        {
            'key': 'multipeak', 'label': 'multipeak', 'type': 'bool',
            'description': 'Defines if the cohere experiment should use multipeak processing.',
        },
        {
            'key': 'converter_ver', 'label': 'converter version', 'type': 'int', 'visibility': 'False',
            'description': 'Converter version.',
        },
    ]

MANDATORY = ['working_dir', 'experiment_id', ]

def get_config_schema():
    return EXP_FIELDS