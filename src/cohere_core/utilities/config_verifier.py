# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.config_verifier
===========================

Verification of configuration parameters.
"""

from pathlib import Path
from cohere_core.controller.op_flow import sub_triggers
import cohere_core.utilities.schemas.beam_prep_schema as prep_schema
import cohere_core.utilities.schemas.exp_schema as exp_schema
import cohere_core.utilities.schemas.post_schema as post_schema
import cohere_core.utilities.schemas.recon_schema as recon_schema
import cohere_core.utilities.schemas.standard_prep_schema as st_prep_schema
import cohere_core.utilities.schemas.mp_schema as mp_schema

ERROR = 2
WARNING = 1
OK = 0


def check_type(value, schema_type):
    match schema_type:
        case 'str':
            if isinstance(value, str):
                return True
        case 'int':
            if isinstance(value, int):
                return True
        case 'float':
            if isinstance(value, float):
                return True
        case 'bool':
            if isinstance(value, bool):
                return True
        case 'list':
            if isinstance(value, list):
                return True
        case 'dict':
            if isinstance(value, dict):
                return True
        case 'file':
            if isinstance(value, str):
                return True
        case 'dir':
            if isinstance(value, str):
                return True
        case 'choice':
            # return True always, will be checked later
            return True
        case 'trigger':
            if isinstance(value, list):
                return True
        case _:
            raise NotImplementedError(f"the type '{schema_type}' defined in schema for parameter is not supported in verifier")
    return False


def validate_file(value, param):
    path = Path(value)
    if not path.is_file():
        return (f'the file {value} for parameter {param} does not exist', WARNING)
    else:
        return ('', OK)


def validate_dir(value, param):
    path = Path(value)
    if not path.is_dir():
        return (f'the directory {value} for parameter {param} does not exist', WARNING)
    else:
        return ('', OK)


def validate_choice(value, params_def, param):
    choices = params_def.get('choices', None)
    if choices is None:
        # pass for now as auto_choices will be allowed
        pass
    elif not value in choices:
        return (f'the parameter {param} has a value of {value}  which is not one of the supported choices: {choices}', ERROR)
    return ('', OK)


def validate_list(value, params_def, param):
    # check subtype, i.e. what type should be elements in the list
    element_type = params_def.get('element_type', None)
    if element_type is None:
        print(f'Info: schema does not define element_type for parameter {param}')
    else:
        ver_elements = [check_type(v, element_type) for v in value]  # validate type of each element in list
        if not any(ver_elements):  # all are False, so it could be alternate type allowed
            alt_element_types = params_def.get('alt_element_types', None)
            if alt_element_types is None:
                # it's an error because the elements in the list do not match
                # the element_type, and no alt_element_types is defined
                return (f'parameter {param} should be a list of {element_type}', ERROR)
            else:  # check for the alt types
                for alt_type in alt_element_types:
                    ver_elements = [check_type(v, alt_type) for v in value]
                    if all(ver_elements):
                        break  # all elements are of one of that type

                if not all(ver_elements):  # check the final
                    return (f'parameter {param} should be a list of {element_type} or {alt_element_types}', ERROR)
    return ('', OK)


def validate_trigger(value, param):
    # trigger should be a list of int or list of lists of int
    if isinstance(value, list):
        if all([check_type(v, 'int') for v in value]):
            return ('', OK)
        elif param in sub_triggers.values():    # the trigger can be configured as subtrigger
            if all([check_type(v, 'list') for v in value]) and \
                    all(isinstance(item, int) for sublist in value for item in sublist):
                return ('', OK)
        else :
            return (f'Trigger {param} should be a list of int, or for subtrigger a list of triggers, i.e. lists of int', ERROR)
    else:
        return (f'Trigger {param} is not a list', ERROR)


def check_param(value, params_def):
    er_msg, res = '', OK
    schema_type = params_def.get('type', None)
    param = params_def['key']
    if schema_type is None:
        if 'placeholder' in params_def.keys():
            schema_type = 'list'
        else:
            print(f'Info: Parameter {param} type is not defined in schema, and is not a placeholder' )

    if schema_type == 'trigger':
        er_msg, res = validate_trigger(value, param)
    elif check_type(value, schema_type): # type is ok, but for some types there are additional checks
        if schema_type == 'list':
            er_msg, res = validate_list(value, params_def, param)
        elif schema_type == 'file':
            er_msg, res = validate_file(value, param)
        elif schema_type == 'dir':
            er_msg, res = validate_dir(value, param)
        elif schema_type == 'choice':
            er_msg, res = validate_choice(value, params_def, param)
    else:   # check for alternate type, later add alternate element type if list
        alt_types = params_def.get('alt_types', None)
        if alt_types is None:
            er_msg, res = (f'Parameter {param} should be of type {schema_type}', ERROR)
        else:
            for at in alt_types:
                validated = check_type(value, at)
                if validated:
                    break
            if not validated:
                er_msg, res = (f'Parameter {param} should be of type {schema_type} or {alt_types}', ERROR)
    return er_msg, res


def group_no_key(params_def, conf_map):
    if 'group' not in params_def:
        # parameter is not a member of any group
        return False

    # group can be a parameter or parameter and choice value separated by "."
    group = params_def['group'].split('.')
    if group[0] not in conf_map:
        # the group key member is not in conf_map
        return True
    elif len(group) == 2 and conf_map[group[0]] != group[1]:
        # the group key member is in conf_map but the choice which is defined in schema as group[1] is
        # not what the value of the key group member is in conf_map
        return True
    else:
        return False


def verify_types(conf_schema, conf_map):
    """
    Verifies parameters types.

    :param conf_schema: dict defining the parameters types
    :param conf_map: dict with the parameters to verify.
        It defaults to None. 
    :return: string containing the error message or empty string is successful
    """
    # schema can be a dict or a list of dictionaries.
    # find if this is the case and flatten to one dict
    if isinstance(conf_schema, dict):
        # case of dict with lists
        params_def = []
        for sd in conf_schema.values():
            params_def.extend(sd)
    elif isinstance(conf_schema, list):
        # case where there is one list
        params_def = conf_schema

    params_def = {pd['key']: pd for pd in params_def}
    errors = ''
    # For each parameter in conf_map dict find definition and check if the type is ok.
    not_in_schema = []
    for k, v in conf_map.items():
        if k in params_def.keys():
            # check if parameter is part of group which key is not in conf_map and skip check if so
            if group_no_key(params_def[k], conf_map):
                continue
            err_msg, er = check_param(v, params_def[k]) # err = 2 if error, 1 if warning, 0 if ok
            if er > 0:
                errors += '\n' + err_msg
                # print(err_msg)
                # if er == 2:  # The parameter is of wrong type
                #     return err_msg
        else:
           not_in_schema.append(k)
    if len(not_in_schema) > 0:
        pass
        # only print warning to allow new parameters
        # print(f'Warning: the following parameters are not in the schema: {not_in_schema}')
    return errors


def verify_params(config_name, conf_map):
    """
    Verifies if mandatory parameters are configured.
    Verifies if group parameters are configured in case group key is configured or set to specific choice.

    :param config_schema: dict defining the parameters types
    :param conf_map: dict with the parameters to verify.
        It defaults to None.
    :return: string containing the error message or empty string is successful
    """
    match config_name:
        case 'config':
            schema_file = exp_schema
        case 'config_disp':
            schema_file = post_schema
        case 'config_rec':
            schema_file = recon_schema
        case 'config_data':
            schema_file = st_prep_schema
        case 'config_prep':
            # schema_file = prep_schema
            return ''   # no mandatory params or groups
        case 'config_instr':
            return ''   # no check as the parameters are checked in code
        case 'config_mp':
            schema_file = mp_schema
        case _:
            raise NotImplementedError(f"Parameter presence check not implemented for {config_name}.")
    errors = ''
    try:
        mandatory_params = schema_file.MANDATORY
    except:
        mandatory_params = None
    if mandatory_params is not None:
        for param in mandatory_params:
            if param not in conf_map.keys():
                errors += '\n' + (f'Mandatory parameter {param} is not defined in {config_name}')

    try:
        mandatory_groups = schema_file.MANDATORY_GROUPS
    except:
        mandatory_groups = None
    if mandatory_groups is not None:
        for group in mandatory_groups:
            alt_params = group.replace(" ", "").split(',')
            group_presence = [e in conf_map.keys() for e in alt_params]
            if not any(group_presence):
                errors += '\n' + (f'One of the parameters: {group} must be defined in {config_name}')

    try:
        groups = schema_file.GROUPS
    except:
        groups = None
    if groups is not None:
 #       for group in groups:
            for k, v in groups.items():
                if k in conf_map.keys():
                    if isinstance(v, dict):
                        # case where the key parameter is a choice and the choice dictates required params
                        mandatory = v[conf_map[k]]
                    elif isinstance(v, list):
                        # case where the key parameter's presence dictates required params (ex. feature)
                        mandatory = v
                    for param in mandatory:
                        if param not in conf_map.keys():
                            errors += '\n' + (f'Parameter {param} is not defined in {config_name}')

    return errors

