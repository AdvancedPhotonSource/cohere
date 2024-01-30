import numpy as np
import re

#
algs = {'ER': ('er', 'modulus'),
        'HIO': ('hio', 'modulus'),
        'ERpc': ('er', 'pc_modulus'),
        'HIOpc': ('hio', 'pc_modulus'),
        'SF' : ('new_alg', 'pc_modulus'),
        }

# this map keeps the names of triggers that can be configured as sub-trigger, i.e. be a trigger for the iteration span
# defined by preceding algorithm. The key is the trigger name and value is the mnemonic. The mnemonic is used in the
# configuration.
sub_triggers = {'shrink_wrap_trigger': 'SW',
             'phm_trigger': 'PHM',
             'lowpass_filter_trigger': 'LPF'}

def get_algs():
    return algs


def get_alg_rows(s, pc_conf_start):
    seq = []
    accum_iter = 0

    def parse_entry(ent, accum_iter):
        r_e = ent.split('*')
        seq.append([int(r_e[0]), r_e[1], accum_iter])
        accum_iter += int(r_e[0])
        return accum_iter

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
    rows = {}
    sub_rows = {}
    row = np.zeros(iter_no, dtype=int)
    fs = set([i for sub in algs.values() for i in sub])
    for f in fs:
        rows[f] = row.copy()
    # for each possible subtrigger add entry
    for f in sub_triggers.values():
        sub_rows[f] = []
    i = 0
    pc_start = None
    for entry in seq:
        repeat = entry[0]
        funs = entry[1].split('.')
        if funs[0] not in algs:
            return 'undefined algorithm ' + funs[0]
        row_keys = algs[funs[0]]
        for row_key in row_keys:
            rows[row_key][i:i+repeat] = 1
            if 'pc' in row_key and pc_start is None:
                pc_start = i
        # find sub-triggers
        for row_key in funs[1:]:
            match = re.match(r"([A-Z]+)([0-9]+)", row_key, re.I)
            if match:
                (feature, idx) = match.groups(0)
                # sub_rows[trigs[feature]].append((entry[2], entry[0] + entry[2], idx))
                sub_rows[feature].append((entry[2], entry[0] + entry[2], idx))
        i += repeat
    return rows, sub_rows, iter_no, pc_start


def trigger_row(trig, iter_no, row=None):
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

    parsed_seq = get_alg_rows(params['algorithm_sequence'], pc_conf_start)
    if type(parsed_seq) == str:
        return None, None
    alg_rows, sub_trig_rows, iter_no, pc_start = get_alg_rows(params['algorithm_sequence'], pc_conf_start)
    flow_arr = np.zeros((len(flow_items_list), iter_no), dtype=int)

    sub_feats = {}
    is_res = False
    # special case; during lowpass filter a lpf shrink wrap is applied, and shrink wrap suppressed
    apply_sw_row = np.ones(iter_no, dtype=int)
    # last iteration when lowpass filter is applied
    last_lpf = 0
    for i in range(len(flow_items_list)):
        flow_item = flow_items_list[i]
        if flow_item == 'next' or flow_item == 'to_reciprocal_space' or flow_item == 'to_direct_space':
            flow_arr[i, :] = 1
        elif flow_item == 'lowpass_filter_trigger':
            if first_run \
                    and 'lowpass_filter_trigger' in params \
                    and len(params['lowpass_filter_trigger']) == 3 \
                    and type(params['lowpass_filter_trigger'][0]) == int:
                flow_arr[i] = trigger_row(params['lowpass_filter_trigger'], iter_no)
                apply_sw_row[params['lowpass_filter_trigger'][0]:params['lowpass_filter_trigger'][2]] = 0
                is_res = True
        elif flow_item == 'reset_resolution':
            if is_res:
                flow_arr[i] = trigger_row([params['lowpass_filter_trigger'][-1],], iter_no)
            if last_lpf > 0:
                flow_arr[i][last_lpf] = 1
        elif flow_item == 'shrink_wrap_trigger':
            if 'shrink_wrap_trigger' in params and type(params['shrink_wrap_trigger'][0]) == int:
                flow_arr[i] = trigger_row(params['shrink_wrap_trigger'], iter_no) * apply_sw_row
        elif flow_item == 'phm_trigger':
            if first_run and 'phm_trigger' in params and type(params['phm_trigger'][0]) == int:
                flow_arr[i] = trigger_row(params['phm_trigger'], iter_no)
        elif flow_item == 'new_func_trigger':
            if 'new_func_trigger' in [algs]:
                flow_arr[i] = trigger_row(params['new_func_trigger'], iter_no)
        elif flow_item == 'pc_trigger':
            if pc_start is not None:
                pc_interval = params['pc_interval']
                pc_trigger = [pc_start, pc_interval]
                flow_arr[i] = trigger_row(pc_trigger, iter_no)
                pc_row = i
        elif flow_item == 'set_prev_pc_trigger':
            if pc_start is not None:
                flow_arr[i, : -1] = flow_arr[pc_row, 1:]
        elif flow_item in alg_rows.keys():
            flow_arr[i] = alg_rows[flow_item]
        elif flow_item == 'twin_trigger':
            if first_run and 'twin_trigger' in params:
                flow_arr[i] = trigger_row(params['twin_trigger'], iter_no)
        elif flow_item == 'average_trigger':
            if 'average_trigger' in params and curr_gen is not None and curr_gen == params['ga_generations'] -1:
                flow_arr[i] = trigger_row(params['average_trigger'], iter_no)
        elif flow_item == 'progress_trigger':
            if 'progress_trigger' in params:
                flow_arr[i] = trigger_row(params['progress_trigger'], iter_no)
                flow_arr[i][-1] = 1
        elif flow_item == 'switch_peaks':
            if 'switch_peak_trigger' in params:
                flow_arr[i] = trigger_row(params['switch_peak_trigger'], iter_no)
                flow_arr[i][-1] = 1

        # Determine features based on sub-triggers
        if flow_item in sub_triggers.keys():
            mne = sub_triggers[flow_item]
            if len(sub_trig_rows[sub_triggers[flow_item]]):
                sub_trig_row = np.zeros(iter_no, dtype=int)
                sub_feats_row = np.zeros(iter_no, dtype=int)
                for (b, e, idx) in sub_trig_rows[mne]:
                    index = int(idx)
                    sub_feats_row[b:e] = index + 1
                    # special case for lowpass filter feature that suppresses shrink wrap
                    if flow_item == 'lowpass_filter_trigger':
                        apply_sw_row[b:e] = 0
                        last_lpf = e
                    trigger = params[flow_item][index].copy()
                    trigger[0] += b
                    if len(trigger) == 2:
                        trigger.append(e)
                    elif len(trigger) == 3:
                        trigger[2] = min(e, trigger[0] + trigger[2])
                    sub_trig_row = trigger_row(trigger, iter_no, sub_trig_row)
                flow_arr[i] = sub_trig_row
                sub_feats_row = sub_feats_row * sub_trig_row
                # suppress the shrink wrap during lowpass filtering
                if flow_item == 'shrink_wrap_trigger':
                    sub_feats_row *= apply_sw_row
                    flow_arr[i] *= apply_sw_row
                sub_feats[mne] = (sub_feats_row, sub_trig_rows[mne])
    return pc_start is not None, flow_arr, sub_feats
