import numpy as np


algs = {'ER': ('er', 'modulus'),
        'HIO': ('hio', 'modulus'),
        'ERpc': ('er', 'pc_modulus'),
        'HIOpc': ('hio', 'pc_modulus'),
        'SF' : ('new_alg', 'pc_modulus')
        }


def get_algs():
    return algs


def get_alg_rows(s, pc_conf_start):
    seq = []
    def parse_entry(ent):
        r_e = ent.split('*')
        seq.append([int(r_e[0]), r_e[1]])

    if pc_conf_start is None:  # no pc in this
        # this is kind of hackish, but will be replaced by sequence for each generation
        if s =='ERpc':
            s = 'ER'
        if s == 'HIOpc':
            s = 'HIO'
    elif not pc_conf_start:    # GA case, the coherence will start at first iteration
        if s == 'ER':
            s = 'ERpc'
        if s == 'HIO':
            s = 'HIOpc'

    s = s.replace(' ','')
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
                    parse_entry(group_entry)
            i += 1
        else:
            parse_entry(entry)
            i += 1
    iter_no = sum([e[0] for e in seq])
    rows = {}
    row = np.zeros(iter_no, dtype=int)
    fs = set([i for sub in algs.values() for i in sub])
    for f in fs:
        rows[f] = row.copy()
    i = 0
    pc_start = None
    for entry in seq:
        repeat = entry[0]
        row_keys = algs[entry[1]]
        for row_key in row_keys:
            rows[row_key][i:i+repeat] = 1
            if 'pc' in row_key and pc_start is None:
                pc_start = i
        i += repeat
    return rows, iter_no, pc_start


def trigger_row(trig, iter_no):
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
    return row


def get_flow_arr(params, flow_items_list, curr_gen=None, first_run=False):
    # get information about GA/pc from config_map
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

    alg_rows, iter_no, pc_start = get_alg_rows(params['algorithm_sequence'], pc_conf_start)
    flow_arr = np.zeros((len(flow_items_list), iter_no), dtype=int)

    is_res = False
    for i in range(len(flow_items_list)):
        if flow_items_list[i] == 'next' or flow_items_list[i] == 'to_reciprocal_space' or flow_items_list[
            i] == 'to_direct_space':
            flow_arr[i, :] = 1
        elif flow_items_list[i] == 'resolution_trigger':
            if first_run and 'resolution_trigger' in params and len(params['resolution_trigger']) == 3:
                flow_arr[i] = trigger_row(params['resolution_trigger'], iter_no)
                is_res = True
        elif flow_items_list[i] == 'reset_resolution':
            if is_res:
                flow_arr[i] = trigger_row([params['resolution_trigger'][-1],], iter_no)
        elif flow_items_list[i] == 'shrink_wrap_trigger':
            if 'shrink_wrap_trigger' in params:
                flow_arr[i] = trigger_row(params['shrink_wrap_trigger'], iter_no)
        elif flow_items_list[i] == 'phase_support_trigger':
            if first_run and 'phase_support_trigger' in params:
                flow_arr[i] = trigger_row(params['phase_support_trigger'], iter_no)
        elif flow_items_list[i] == 'new_func_trigger':
            if 'new_func_trigger' in [algs]:
                flow_arr[i] = trigger_row(params['new_func_trigger'], iter_no)
        elif flow_items_list[i] == 'pc_trigger':
            if pc_start is not None:
                pc_interval = params['pc_interval']
                pc_trigger = [pc_start, pc_interval]
                flow_arr[i] = trigger_row(pc_trigger, iter_no)
                pc_row = i
        elif flow_items_list[i] == 'set_prev_pc_trigger':
            if pc_start is not None:
                flow_arr[i, : -1] = flow_arr[pc_row, 1:]
        elif flow_items_list[i] in alg_rows.keys():
            flow_arr[i] = alg_rows[flow_items_list[i]]
        elif flow_items_list[i] == 'twin_trigger':
            if first_run and 'twin_trigger' in params:
                flow_arr[i] = trigger_row(params['twin_trigger'], iter_no)
        elif flow_items_list[i] == 'average_trigger':
            if 'average_trigger' in params and curr_gen is not None and curr_gen == params['ga_generations'] -1:
                flow_arr[i] = trigger_row(params['average_trigger'], iter_no)
        elif flow_items_list[i] == 'progress_trigger':
            if 'progress_trigger' in params:
                flow_arr[i] = trigger_row(params['progress_trigger'], iter_no)
                flow_arr[i][-1] = 1

    return pc_start is not None, flow_arr
