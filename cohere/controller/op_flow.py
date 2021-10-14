import numpy as np


def get_alg_seq(conf_algorithm_sequence):
    def add_list(algorithm_sequence, repeat, subseq):
        l = []
        for seq in subseq:
            alg = seq[0]
            alg_repeat = seq[1]
            templ = [alg] * alg_repeat
            l.extend(templ)
        algorithm_sequence.extend((repeat * l))

    algorithm_sequence = []
    for t in conf_algorithm_sequence:
        if type(t[0]) == int:
            add_list(algorithm_sequence, t[0], t[1:])
    return (algorithm_sequence)


def trigger_row(trig, iter_no):
    row = np.zeros(iter_no, dtype=int)
    if len(trig) == 1:
        trig_iter = trig[0]
        if trig_iter < 0:
            trig_iter += iter_no
        row[trig_iter] = 1
    else:
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


def algorithm_row(algorithm, algorithm_sequence):
    row = np.zeros(len(algorithm_sequence), dtype=int)
    for i in range(len(algorithm_sequence)):
        if algorithm.upper() == algorithm_sequence[i]:
            row[i] = 1
    return row


def get_flow_arr(params, flow_items_list, curr_gen=None, first_run=False):
    # tha params hold the parsed values for the parameters, not triggers or algorithm sequence
    # the triggers and algorithm sequence are parsed in this script which determines the functions
    success, config_map = params.read_config()
    algorithm_sequence = get_alg_seq(config_map.algorithm_sequence)
    iter_no = len(algorithm_sequence)
    flow_arr = np.zeros((len(flow_items_list), iter_no), dtype=int)

    pcdi_start = None
    for i in range(len(flow_items_list)):
        if flow_items_list[i] == 'next' or flow_items_list[i] == 'to_reciprocal_space' or flow_items_list[
            i] == 'to_direct_space':
            flow_arr[i, :] = 1
        elif flow_items_list[i] == 'resolution_trigger' and first_run:
            if config_map.lookup('resolution_trigger') is not None and len(config_map.resolution_trigger) == 3:
                flow_arr[i] = trigger_row(config_map.resolution_trigger, iter_no)
        elif flow_items_list[i] == 'shrink_wrap_trigger':
            if config_map.lookup('shrink_wrap_trigger') is not None:
                flow_arr[i] = trigger_row(config_map.shrink_wrap_trigger, iter_no)
        elif flow_items_list[i] == 'phase_support_trigger' and first_run:
            if config_map.lookup('phase_support_trigger') is not None:
                flow_arr[i] = trigger_row(config_map.phase_support_trigger, iter_no)
        elif flow_items_list[i] == 'new_func_trigger':
            if config_map.lookup('new_func_trigger') is not None:
                flow_arr[i] = trigger_row(config_map.new_func_trigger, iter_no)
        elif flow_items_list[i] == 'pcdi_trigger':
            if config_map.lookup('pcdi_trigger') is not None:
                calculated_first_run = first_run
                if curr_gen is not None:
                    if config_map.lookup('gen_pcdi_start') is None:
                        gen_pcdi_start = 0
                    else:
                        gen_pcdi_start = config_map.gen_pcdi_start
                    if curr_gen < gen_pcdi_start:
                        calculated_first_run = None
                    elif curr_gen == gen_pcdi_start:
                        calculated_first_run = True
                    else:
                        calculated_first_run = False
                if calculated_first_run is None:
                    pcdi_start = None
                else:
                    flow_arr[i] = trigger_row(config_map.pcdi_trigger, iter_no)
                    if calculated_first_run:
                        pcdi_start = config_map.pcdi_trigger[0]
                    else:
                        pcdi_start = 0
                    pcdi_row = i
        elif flow_items_list[i] == 'pcdi_modulus':
            if pcdi_start is not None:
                flow_arr[i, pcdi_start:] = 1
        elif flow_items_list[i] == 'modulus':
            if pcdi_start is not None:
                flow_arr[i, : pcdi_start] = 1
            else:
                flow_arr[i, :] = 1
        elif flow_items_list[i] == 'set_prev_pcdi_trigger':
            if pcdi_start is not None:
                flow_arr[i, : -1] = flow_arr[pcdi_row, 1:]
        elif flow_items_list[i] == 'er' or flow_items_list[i] == 'hio' or flow_items_list[i] == 'new_alg':
            flow_arr[i] = algorithm_row(flow_items_list[i], algorithm_sequence)
        elif flow_items_list[i] == 'twin_trigger' and first_run:
            if config_map.lookup('twin_trigger') is not None:
                flow_arr[i] = trigger_row(config_map.twin_trigger, iter_no)
        elif flow_items_list[i] == 'average_trigger':
            if config_map.lookup('average_trigger') is not None and curr_gen == config_map.generations -1:
                flow_arr[i] = trigger_row(config_map.average_trigger, iter_no)
        elif flow_items_list[i] == 'progress_trigger':
            if config_map.lookup('progress_trigger') is not None:
                flow_arr[i] = trigger_row(config_map.progress_trigger, iter_no)

    return pcdi_start is not None, flow_arr

#
# conf = ut.read_config('/Users/bfrosik/test/a_54-66/conf/config_rec')
# a = get_flow_arr(conf, True)
## print(a)
