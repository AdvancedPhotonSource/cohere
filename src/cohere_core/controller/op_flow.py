# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import re

# This dict maps the mnemonic used when defining algorithm sequence parameter to the four steps of
# phase retrieval algorithm functions.

algs = {'ER': ('to_reciprocal_space', 'modulus', 'to_direct_space', 'er'),
        'HIO': ('to_reciprocal_space', 'modulus', 'to_direct_space', 'hio'),
        'ERpc': ('to_reciprocal_space', 'pc_modulus', 'to_direct_space', 'er'),
        'HIOpc': ('to_reciprocal_space', 'pc_modulus', 'to_direct_space', 'hio'),
        'SF': ('to_reciprocal_space', 'modulus', 'to_direct_space', 'sf'),
        'RAAR': ('to_reciprocal_space', 'modulus', 'to_direct_space', 'raar')
        }

# This map keeps the names of triggers that can be configured as sub-trigger, i.e. be a trigger for the iteration span
# defined by preceding algorithm. The key is the trigger name and value is the mnemonic. The mnemonic is used in the
# configuration.
sub_triggers = {'SW' : 'shrink_wrap_trigger',
             'PHC' : 'phc_trigger'}

# This list contains triggers that will be active at the last iteration defined by trigger, despite
# not being a trigger calculated by the step formula.
# It applies to sub-triggers, setting the last iteration to that of sub-trigger.
last_iter_op_triggers = ['progress_trigger',
                         'switch_peaks_trigger',
                         'switch_resampling_trigger']

def get_alg_rows(s, pc_conf_start):
    """
    Parses algorithm sequence string into structures being: algorithm rows, and sub-trigger operations info.

    :param s: str
        algorithm sequence
    :param pc_conf_start: boolean or None
        if None, no partial coherence is scheduled
        if True, the configured partial coherence will be scheduled
        if False, the partial coherence started ago (in GA case) and will continue here
    :return: tuple
        rows : ndarry
             ndarray that depicts algorithms (modulus, pc_modulus, hio, er) operations
        sub_rows : dict
            dictionary with entries of k : v, where
            k is the trigger name that is being configured as sub-triggers
            v is a list of sub-trigger operations
        iter_no : int
            number of iterations
        pc_start : None or int
            starting iteration of partial coherence if any
    """
    seq = []
    accum_iter = 0

    def parse_entry(ent, accum_iter):
        # parses elementary part of the algorithm sequence
        r_e = ent.split('*')
        seq.append([int(r_e[0]), r_e[1], accum_iter])
        accum_iter += int(r_e[0])
        return accum_iter

    s = s.replace(' ', '')
    entries = s.split('+')
    i = 0
    while i < len(entries):
        entry = entries[i]
        if '(' in entry:
            group = []
            rep_entry = entry.split('(')
            repeat = int(rep_entry[0][:-1])
            group.append(rep_entry[1])
            i += 1
            group_entry = entries[i]
            while ')' not in group_entry:
                group.append(group_entry)
                i += 1
                group_entry = entries[i]
            group.append(group_entry[:-1])
            for _ in range(repeat):
                for group_entry in group:
                    accum_iter = parse_entry(group_entry, accum_iter)
            i += 1
        else:
            accum_iter = parse_entry(entry, accum_iter)
            i += 1
    iter_no = sum([e[0] for e in seq])
    alg_rows = {}
    sub_rows = {}
    row = np.zeros(iter_no, dtype=int)
    fs = set([i for sub in algs.values() for i in sub])
    for f in fs:
        alg_rows[f] = row.copy()
    i = 0
    pc_start = None
    for entry in seq:
        repeat = entry[0]
        funs = entry[1].split('.')
        if funs[0] not in algs:
            msg = f'algorithm {funs[0]} is not defined in op_flow.py file, algs dict.'
            raise NameError(msg)
        # the pc will not be executed if pc_conf_start is None
        # this code will be revised after each generation has separate config
        if pc_conf_start is None:
            if funs[0].endswith('pc'):
                funs[0] = funs[0][:-2]
        elif not pc_conf_start:
            if not funs[0].endswith('pc'):
                funs[0] = funs[0] + 'pc'

        row_keys = algs[funs[0]]
        for row_key in row_keys:
            alg_rows[row_key][i:i+repeat] = 1
            if 'pc' in row_key and pc_start == None:
                if pc_conf_start == True:
                    pc_start = i
                else:
                    pc_start = 1
        # find sub-triggers
        for row_key in funs[1:]:
            match = re.match(r"([A-Z]+)([0-9]+)", row_key, re.I)
            if match:
                (trig_op, idx) = match.groups(0)
                sub_t = sub_triggers[trig_op]
                if trig_op not in sub_triggers.keys():
                    msg = f'the sub-trigger {trig_op} must be defined in op_flow.py file, sub_triggers dict.'
                    raise NameError(msg)
                if sub_t not in sub_rows:
                    sub_rows[sub_t] = []
                sub_rows[sub_t].append((entry[2], entry[0] + entry[2], idx))
        i += repeat

    return alg_rows, sub_rows, iter_no, pc_start


def fill_trigger_row(trig, iter_no, last_trig, row=None):
    """
    This functions creates ndarray that depicts triggered operations for a given trigger.

    :param trig: list
        a list with 1, 2, or 3 elements defining trigger
    :param iter_no: int
        total number of iterations
    :param row: ndarray
        if given, the row will be used to fill the trigger
    :return:
    """
    if row is None:
        row = np.zeros(iter_no, dtype=int)
    if len(trig) ==1:
        trig_iter = trig[0]
        if trig_iter < 0:
            trig_iter += iter_no
        row[trig_iter] = 1
    else:   # trig is list
        trig_start = trig[0]
        if trig_start < 0:
            trig_start += iter_no
        step = trig[1]
        if len(trig) == 3:
            trig_stop = trig[2]
            if trig_stop < 0:
                trig_stop += iter_no
        else:
            trig_stop = iter_no
        for it in range(trig_start, trig_stop, step):
            row[it] = 1
        if last_trig:
            row[trig_stop - 1] = 1
    return row


def fill_sub_trigger_row(sub_iters, sub_trigs, iter_no, last_trig):
    """
    Based on iterations allocated to sub-triggers and sub-triggers definitions this functions
    creates ndarray that depicts triggered operations.

    :param sub_iters: list
        contains entry for each sub-trigger
        the entry consisting of starting and ending iterations where the sub-trigger
        is active and index value (+1) that specifies sub-trigger.
    :param sub_trigs: list
        list of sub-trigger, defined in configuration
    :param iter_no: int
        total number of iterations
    :return: ndarray
        array of int, value of zero meaning no trigger operation in this iteration
        value greater than zero meaning the sub-trigger operation related to the value
        will be triggered in this iteration
    """
    # create array indicating triggered operation (1) or no action (0) along iterations
    sub_trig_row = np.zeros(iter_no, dtype=int)
    # create array indicating with index which sub-triggered operation may happen in the iterations
    sub_trig_idx_row = np.zeros(iter_no, dtype=int)
    # for each defined sub iteration chunk apply corresponding sub-trigger
    for i, sub_iter in enumerate(sub_iters):
        (b, e, idx) = sub_iter
        index = int(idx)
        sub_trig_idx_row[b:e] = index + 1
        if len(sub_trigs) - 1 < index:
            msg = 'not enough entries in sub-trigger'
            raise RuntimeError(msg)
        trigger = sub_trigs[index].copy()
        trigger[0] += b
        if len(trigger) == 2:
            trigger.append(e)
        elif len(trigger) == 3:
            trigger[2] = min(e, trigger[0] + trigger[2])
            # update the sub_iters
            sub_iters[i] = (b, trigger[2], idx)
        sub_trig_row = fill_trigger_row(trigger, iter_no, last_trig, sub_trig_row)

    return sub_trig_row * sub_trig_idx_row


def get_flow_arr(params, flow_items_list, curr_gen=None):
    # get information about GA and partial coherence from config_map
    # pc_conf_start is None if partial coherence is inactive in this reconstruction
    # it is True if the partial coherence starts in this generation
    # and False if partial coherence started in previous generation and is continued
    if 'pc_interval' in params:
        if curr_gen is None:
            pc_conf_start = True
        else:
            if curr_gen < params['ga_gen_pc_start']:
                pc_conf_start = None
            elif curr_gen == params['ga_gen_pc_start']:
                pc_conf_start = True
            else:
                pc_conf_start = False
    else:
        pc_conf_start = None
    if pc_conf_start is None:
        params.pop('pc_interval', None)

    # parse algorithm sequence to get the algorithm rows and sub-triggers rows, number iterations,
    # and partial coherence starting iteration
    try:
        (alg_rows, sub_iters, iter_no, pc_start) = get_alg_rows(params['algorithm_sequence'], pc_conf_start)
    except:
        return False, None, None

    # do some checks to find if the sequence and configuration are runnable
    # and special cases

    last_lpf = None
    if 'lowpass_filter_trigger' in params:
        if len(params['lowpass_filter_trigger']) < 2:
            print('Low pass trigger misconfiguration error. This trigger should have upper bound.')
            raise
        elif params['lowpass_filter_trigger'][2] >= iter_no:
            print('Low pass trigger misconfiguration error. The upper bound should be less than total iterations.')
            raise
        else:
            last_lpf = params['lowpass_filter_trigger'][2]

    if pc_start is not None:
        if pc_start == 0:
            raise ValueError('partial coherence is configured in first iteration, allow several ER before.')
        else:
            pc_interval = params['pc_interval']
            params['pc_trigger'] = [pc_start, pc_interval]

    # initialize
    sub_trig_op = {}

    # create empty array with the size of number of all functions by number of all iterations
    flow_arr = np.zeros((len(flow_items_list), iter_no), dtype=int)

    # fill the flow array with ones if function should execute in iteration
    for i, flow_item in enumerate(flow_items_list):
        if flow_item == 'next':
        # these functions are executed in each iteration
            flow_arr[i, :] = 1
        elif flow_item in alg_rows.keys():
            # fill out the algorithm rows
            flow_arr[i] = alg_rows[flow_item]
        elif flow_item.endswith('operation'):
            # fill out trigger/sub-trigger operations rows
            # The function name and associated trigger differ in prefix.
            # the function name ends with 'operation', and trigger ends with 'trigger'
            trigger_name = flow_item.replace('operation', 'trigger')
            if trigger_name in params:
                # set the switch last_trig if the trigger should end with operation
                last_trig = trigger_name in last_iter_op_triggers

                # determined in algorithm sequence parsing if the triggered operation is configured
                # with sub-triggers or trigger
                if trigger_name in sub_iters.keys():
                    # may throw exception
                    flow_arr[i] = fill_sub_trigger_row(sub_iters[trigger_name], params[trigger_name], iter_no, last_trig)
                    # special case
                    if flow_item == 'phc_operation':
                        reset = [l[1] for l in list(sub_iters[trigger_name])]
                        flow_arr[i-1][reset] = 1

                    # add entry to sub trigger operation dict with key of the trigger mnemonic
                    # and the value of a list with the row and sub triggers iterations chunks
                    sub_trig_op[trigger_name] = (flow_arr[i], sub_iters[trigger_name])
                else:
                    flow_arr[i] = fill_trigger_row(params[trigger_name], iter_no, last_trig)
                    # special case
                    if flow_item == 'phc_operation':
                        # Assuming phc trigger is configured with upper limit
                        reset_iter = min(iter_no - 1, params[trigger_name][2] + 1)
                        flow_arr[i-1][reset_iter] = 1
        elif flow_item == 'set_prev_pc' and pc_start is not None:
            # set_prev_pc is executed one iteration before pc_trigger
            pc_row = flow_items_list.index('pc_operation')
            flow_arr[i, : -1] = flow_arr[pc_row, 1:]
        elif flow_item == 'reset_resolution' and last_lpf is not None:
            # reset low pass filter (i.e. data set to original) after the last LPF operation
            flow_arr[i][last_lpf] = 1

    return pc_start is not None, flow_arr, sub_trig_op
