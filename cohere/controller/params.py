import pylibconfig2 as cfg
import os


class Params:
    def __init__(self, config_file):
        self.config_file = config_file

    def read_config(self):
        """
        This function gets configuration file. It checks if the file exists and parses it into an object.

        Parameters
        ----------
        config_file : str
            configuration file name, including path

        Returns
        -------
        config_map : Config object
            a Config containing parsed configuration, None if the given file does not exist
        """
        if os.path.isfile(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_map = cfg.Config(f.read())
                    return (True, config_map)
            except Exception as e:
                msg = 'cannot parse configuration file', self.config_file, e
                return (False, msg)
        else:
            msg = 'configuration file', self.config_file, 'does not exist'
            return (False, msg)

    def set_params(self):
        (success, res) = self.read_config()
        if not success:
            return res
        conf = res

        if conf.lookup('algorithm_sequence') is None:
            return ('missing algorithm_sequence parameter in config file')
        if conf.lookup('data_dir') is not None:
            self.data_dir = conf.data_dir
        if conf.lookup('save_dir') is not None:
            self.save_dir = conf.save_dir

        if conf.lookup('init_guess') is not None:
            self.init_guess = conf.init_guess
            if self.init_guess == 'continue':
                if conf.lookup('continue_dir') is not None:
                    self.continue_dir = conf.continue_dir
                else:
                    return ('missing continue_dir parameter in config file')
            elif self.init_guess == 'AI_guess':
                if conf.lookup('AI_threshold') is not None:
                    self.AI_threshold = conf.AI_threshold
                else:
                    self.AI_threshold = conf.shrink_wrap_threshold
                if conf.lookup('AI_sigma') is not None:
                    self.AI_sigma = conf.AI_sigma
                else:
                    self.AI_sigma = conf.shrink_wrap_gauss_sigma
            else:
                self.init_guess = 'random'
        else:
            self.init_guess = 'random'

        # if conf.lookup('cont') is not None and conf.cont is True:
        #     self.cont = True
        #     if conf.lookup('continue_dir') is not None:
        #         self.continue_dir = conf.continue_dir
        #     else:
        #         return ('missing continue_dir parameter in config file')
        # else:
        #     self.cont = False

        if conf.lookup('reconstructions') is not None:
            self.reconstructions = conf.reconstructions
        else:
            self.reconstructions = 1
        if conf.lookup('device') is not None:
            self.device = conf.device
        else:
            self.device = (-1)
        if conf.lookup('hio_beta') is not None:
            self.hio_beta = conf.hio_beta
        else:
            self.hio_beta = 0.9
        if conf.lookup('initial_support_area') is not None:
            self.initial_support_area = conf.initial_support_area
        else:
            self.initial_support_area = (.5, .5, .5)

        self.ga_gen_pc_start = None
        if conf.lookup('ga_generations') is not None:
            self.ga_generations = conf.ga_generations
        else:
            self.ga_generations = 1
        if self.ga_generations > 1:
            if conf.lookup('ga_fast') is not None:
                self.ga_fast = conf.ga_fast
            else:
                self.ga_fast = True
            if conf.lookup('self.metrics') is not None:
                self.metrics = list(conf.ga_metrics)
                if len(self.metrics) == 1:
                    self.metrics = self.metrics * self.ga_generations
                elif len(self.metrics) < self.ga_generations:
                    self.metrics = self.metrics + ['chi', ] * (self.ga_generations - len(self.metrics))
            else:
                self.metrics = ['chi', ] * self.ga_generations

            self.ga_reconstructions = []
            if conf.lookup('ga_cullings') is not None:
                worst_remove_no = list(conf.ga_cullings)
                if len(worst_remove_no) < self.ga_generations:
                    worst_remove_no = worst_remove_no + [0, ] * (self.ga_generations - len(worst_remove_no))
            else:
                worst_remove_no = [0, ] * self.ga_generations
            # calculate how many reconstructions should continue
            reconstructions = self.reconstructions
            for cull in worst_remove_no:
                reconstructions = reconstructions - cull
                if reconstructions <= 0:
                    return 'culled down to 0 reconstructions, check configuration'
                self.ga_reconstructions.append(reconstructions)

            if conf.lookup('ga_shrink_wrap_thresholds') is not None:
                self.ga_shrink_wrap_thresholds = list(conf.ga_shrink_wrap_thresholds)
                if len(self.ga_shrink_wrap_thresholds) == 1:
                    self.ga_shrink_wrap_thresholds = self.ga_shrink_wrap_thresholds * self.ga_generations
                elif len(self.ga_shrink_wrap_thresholds) < self.ga_generations:
                    if conf.lookup('shrink_wrap_threshold') is not None:
                        shrink_wrap_threshold = conf.shrink_wrap_threshold
                    else:
                        shrink_wrap_threshold = .1
                    self.ga_shrink_wrap_thresholds = self.ga_shrink_wrap_thresholds + [shrink_wrap_threshold, ] * (
                    self.ga_generations - len(self.ga_shrink_wrap_thresholds))
            else:
                if conf.lookup('shrink_wrap_threshold') is not None:
                    shrink_wrap_threshold = conf.shrink_wrap_threshold
                else:
                    shrink_wrap_threshold = .1
                self.ga_shrink_wrap_thresholds = [shrink_wrap_threshold, ] * (self.ga_generations)

            if conf.lookup('ga_shrink_wrap_gauss_sigmas') is not None:
                self.ga_shrink_wrap_gauss_sigmas = list(conf.ga_shrink_wrap_gauss_sigmas)
                if len(self.ga_shrink_wrap_gauss_sigmas) == 1:
                    self.ga_shrink_wrap_gauss_sigmas = self.ga_shrink_wrap_gauss_sigmas * self.ga_generations
                elif len(self.ga_shrink_wrap_gauss_sigmas) < self.ga_generations:
                    if conf.lookup('beta') is not None:
                        shrink_wrap_gauss_sigma = conf.shrink_wrap_gauss_sigma
                    else:
                        shrink_wrap_gauss_sigma = 1.0
                    self.ga_shrink_wrap_gauss_sigmas = self.ga_shrink_wrap_gauss_sigmas + [shrink_wrap_gauss_sigma, ] * (
                        self.ga_generations - len(self.ga_shrink_wrap_gauss_sigmas))
            else:
                if conf.lookup('shrink_wrap_gauss_sigma') is not None:
                    shrink_wrap_gauss_sigma = conf.shrink_wrap_gauss_sigma
                else:
                    shrink_wrap_gauss_sigma = 1.0
                self.ga_shrink_wrap_gauss_sigmas = [shrink_wrap_gauss_sigma, ] * (self.ga_generations)

            if conf.lookup('ga_breed_modes') is not None:
                self.breed_modes = list(conf.ga_breed_modes)
                if len(self.breed_modes) == 1:
                    self.breed_modes = self.breed_modes * self.ga_generations
                elif len(self.breed_modes) < self.ga_generations:
                    self.breed_modes = self.breed_modes + ['sqrt_ab', ] * (self.ga_generations - len(self.breed_modes))
            else:
                self.breed_modes = ['none', ] * self.ga_generations

            if conf.lookup('ga_lowpass_filter_sigmas') is not None:
                self.ga_lowpass_filter_sigmas = conf.ga_lowpass_filter_sigmas
                self.low_resolution_generations = len(self.ga_lowpass_filter_sigmas)
            else:
                self.low_resolution_generations = 0

            if self.low_resolution_generations > 0:
                if conf.lookup('low_resolution_alg') is not None:
                    self.low_resolution_alg = conf.ga_low_resolution_alg
                else:
                    self.low_resolution_alg = 'GAUSS'

            if conf.lookup('pc_trigger') is not None:
                if conf.lookup('ga_gen_pc_start') is not None:
                    self.ga_gen_pc_start = conf.ga_gen_pc_start
                else:
                    self.ga_gen_pc_start = 0

        if conf.lookup('twin_trigger') is not None:
            if conf.lookup('twin_halves') is not None:
                self.twin_halves = conf.twin_halves
            else:
                self.twin_halves = (0, 0)

        self.shrink_wrap_gauss_sigma = None
        if conf.lookup('shrink_wrap_trigger') is not None:
            if conf.lookup('shrink_wrap_type') is not None:
                self.shrink_wrap_type = conf.shrink_wrap_type
            else:
                self.shrink_wrap_type = "GAUSS"
            if conf.lookup('shrink_wrap_threshold') is not None:
                self.shrink_wrap_threshold = conf.shrink_wrap_threshold
            else:
                self.shrink_wrap_threshold = 0.1
            if conf.lookup('shrink_wrap_gauss_sigma') is not None:
                self.shrink_wrap_gauss_sigma = conf.shrink_wrap_gauss_sigma
            else:
                self.shrink_wrap_gauss_sigma = 1.0

        if conf.lookup('phase_support_trigger') is not None:
            if conf.lookup('phm_phase_min') is not None:
                self.phm_phase_min = conf.phm_phase_min
            else:
                self.phm_phase_min = -1.57
            if conf.lookup('phm_phase_max') is not None:
                self.phm_phase_max = conf.phm_phase_max
            else:
                self.phm_phase_max = 1.57

        if conf.lookup('new_func_trigger') is not None:
            if conf.lookup('new_param') is not None:
                self.new_param = conf.new_param
            else:
                self.new_param = 1

        if conf.lookup('pc_trigger') is not None:
            self.is_pcdi = True
            if conf.lookup('pc_type') is not None:
                self.pc_type = conf.pc_type
            else:
                self.pc_type = "LUCY"
            if conf.lookup('pc_LUCY_iterations') is not None:
                self.pc_LUCY_iterations = conf.pc_LUCY_iterations
            else:
                self.pc_LUCY_iterations = 20
            if conf.lookup('pc_normalize') is not None:
                self.pc_normalize = conf.pc_normalize
            else:
                self.pc_normalize = True
            if conf.lookup('pc_LUCY_kernel') is not None:
                self.pc_LUCY_kernel = conf.pc_LUCY_kernel
            else:
                self.pc_LUCY_kernel = (16, 16, 16)
        else:
            self.is_pcdi = False

        self.ll_sigmas = None
        self.ll_dets = None
        if conf.lookup('resolution_trigger') is not None and len(conf.resolution_trigger) == 3:
            # linespacing the sigmas and dets need a number of iterations
            # The trigger should have three elements, the last one
            # meaning the last iteration when the low resolution is
            # applied. If the trigger does not have the limit,
            # it is misconfigured, and the trigger will not be
            # active
            ll_iter = conf.resolution_trigger[2]
            if conf.lookup('shrink_wrap_trigger') is not None and conf.lookup('lowpass_filter_sw_sigma_range') is not None:
                # The sigmas are used to find support, if the shrink wrap
                # trigger is not active, it wil not be used
                sigma_range = conf.lowpass_filter_sw_sigma_range
                if len(sigma_range) > 0:
                    first_sigma = sigma_range[0]
                    if len(sigma_range) == 1:
                        last_sigma = self.shrink_wrap_gauss_sigma
                    else:
                        last_sigma = sigma_range[1]
                    self.ll_sigmas = [first_sigma + x * (last_sigma - first_sigma) / ll_iter for x in range(ll_iter)]
            self.ll_dets = None
            if conf.lookup('lowpass_filter_range') is not None:
                det_range = conf.lowpass_filter_range
                if len(det_range) > 0:
                    first_det = det_range[0]
                    if len(det_range) == 1:
                        last_det = 1.0
                    else:
                        last_det = det_range[1]
                    self.ll_dets = [first_det + x * (last_det - first_det) / ll_iter for x in range(ll_iter)]

        return None

