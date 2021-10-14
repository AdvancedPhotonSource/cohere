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
        if conf.lookup('cont') is not None and conf.cont is True:
            self.cont = True
            if conf.lookup('continue_dir') is not None:
                self.continue_dir = conf.continue_dir
            else:
                return ('missing continue_dir parameter in config file')
        else:
            self.cont = False
        if conf.lookup('reconstructions') is not None:
            self.reconstructions = conf.reconstructions
        else:
            self.reconstructions = 1
        if conf.lookup('device') is not None:
            self.device = conf.device
        else:
            self.device = (-1)
        if conf.lookup('beta') is not None:
            self.beta = conf.beta
        else:
            self.beta = 0.9
        if conf.lookup('support_area') is not None:
            self.support_area = conf.support_area
        else:
            self.support_area = (.5, .5, .5)

        self.gen_pcdi_start = None
        if conf.lookup('generations') is not None:
            self.generations = conf.generations
        else:
            self.generations = 1
        if self.generations > 1:
            if conf.lookup('ga_fast') is not None:
                self.ga_fast = conf.ga_fast
            else:
                self.ga_fast = True
            if conf.lookup('self.metrics') is not None:
                self.metrics = list(conf.ga_metrics)
                if len(self.metrics) == 1:
                    self.metrics = self.metrics * self.generations
                elif len(self.metrics) < self.generations:
                    self.metrics = self.metrics + ['chi', ] * (self.generations - len(self.metrics))
            else:
                self.metrics = ['chi', ] * self.generations

            self.ga_reconstructions = []
            if conf.lookup('ga_cullings') is not None:
                worst_remove_no = list(conf.ga_cullings)
                if len(worst_remove_no) < self.generations:
                    worst_remove_no = worst_remove_no + [0, ] * (self.generations - len(worst_remove_no))
            else:
                worst_remove_no = [0, ] * self.generations
            # calculate how many reconstructions should continue
            reconstructions = self.reconstructions
            for cull in worst_remove_no:
                reconstructions = reconstructions - cull
                if reconstructions <= 0:
                    return 'culled down to 0 reconstructions, check configuration'
                self.ga_reconstructions.append(reconstructions)

            if conf.lookup('ga_support_thresholds') is not None:
                self.ga_support_thresholds = list(conf.ga_support_thresholds)
                if len(self.ga_support_thresholds) == 1:
                    self.ga_support_thresholds = self.ga_support_thresholds * self.generations
                elif len(self.ga_support_thresholds) < self.generations:
                    if conf.lookup('support_threshold') is not None:
                        support_threshold = conf.support_threshold
                    else:
                        support_threshold = .1
                    self.ga_support_thresholds = self.ga_support_thresholds + [support_threshold, ] * (
                    self.generations - len(self.ga_support_thresholds))
            else:
                if conf.lookup('support_threshold') is not None:
                    support_threshold = conf.support_threshold
                else:
                    support_threshold = .1
                self.ga_support_thresholds = [support_threshold, ] * (self.generations)

            if conf.lookup('ga_support_sigmas') is not None:
                self.ga_support_sigmas = list(conf.ga_support_sigmas)
                if len(self.ga_support_sigmas) == 1:
                    self.ga_support_sigmas = self.ga_support_sigmas * self.generations
                elif len(self.ga_support_sigmas) < self.generations:
                    if conf.lookup('beta') is not None:
                        support_sigma = conf.support_sigma
                    else:
                        support_sigma = 1.0
                    self.ga_support_sigmas = self.ga_support_sigmas + [support_sigma, ] * (
                        self.generations - len(self.ga_support_sigmas))
            else:
                if conf.lookup('support_sigma') is not None:
                    support_sigma = conf.support_sigma
                else:
                    support_sigma = 1.0
                self.ga_support_sigmas = [support_sigma, ] * (self.generations)

            if conf.lookup('ga_breed_modes') is not None:
                self.breed_modes = list(conf.ga_breed_modes)
                if len(self.breed_modes) == 1:
                    self.breed_modes = self.breed_modes * self.generations
                elif len(self.breed_modes) < self.generations:
                    self.breed_modes = self.breed_modes + ['sqrt_ab', ] * (self.generations - len(self.breed_modes))
            else:
                self.breed_modes = ['none', ] * self.generations

            if conf.lookup('ga_low_resolution_sigmas') is not None:
                self.ga_low_resolution_sigmas = conf.ga_low_resolution_sigmas
                self.low_resolution_generations = len(self.ga_low_resolution_sigmas)
            else:
                self.low_resolution_generations = 0

            if self.low_resolution_generations > 0:
                if conf.lookup('low_resolution_alg') is not None:
                    self.low_resolution_alg = conf.ga_low_resolution_alg
                else:
                    self.low_resolution_alg = 'GAUSS'

            if conf.lookup('pcdi_trigger') is not None:
                if conf.lookup('gen_pcdi_start') is not None:
                    self.gen_pcdi_start = conf.gen_pcdi_start
                else:
                    self.gen_pcdi_start = 0

        if conf.lookup('twin_trigger') is not None:
            if conf.lookup('twin_halves') is not None:
                self.twin_halves = conf.twin_halves
            else:
                self.twin_halves = (0, 0)

        self.support_sigma = None
        if conf.lookup('shrink_wrap_trigger') is not None:
            if conf.lookup('shrink_wrap_type') is not None:
                self.shrink_wrap_type = conf.shrink_wrap_type
            else:
                self.shrink_wrap_type = "GAUSS"
            if conf.lookup('support_threshold') is not None:
                self.support_threshold = conf.support_threshold
            else:
                self.support_threshold = 0.1
            if conf.lookup('support_sigma') is not None:
                self.support_sigma = conf.support_sigma
            else:
                self.support_sigma = 1.0

        if conf.lookup('phase_support_trigger') is not None:
            if conf.lookup('phase_min') is not None:
                self.phase_min = conf.phase_min
            else:
                self.phase_min = -1.57
            if conf.lookup('phase_max') is not None:
                self.phase_max = conf.phase_max
            else:
                self.phase_max = 1.57

        if conf.lookup('new_func_trigger') is not None:
            if conf.lookup('new_param') is not None:
                self.new_param = conf.new_param
            else:
                self.new_param = 1

        if conf.lookup('pcdi_trigger') is not None:
            self.is_pcdi = True
            if conf.lookup('partial_coherence_type') is not None:
                self.partial_coherence_type = conf.partial_coherence_type
            else:
                self.partial_coherence_type = "LUCY"
            if conf.lookup('partial_coherence_iteration_num') is not None:
                self.partial_coherence_iteration_num = conf.partial_coherence_iteration_num
            else:
                self.partial_coherence_iteration_num = 20
            if conf.lookup('partial_coherence_normalize') is not None:
                self.partial_coherence_normalize = conf.partial_coherence_normalize
            else:
                self.partial_coherence_normalize = True
            if conf.lookup('partial_coherence_roi') is not None:
                self.partial_coherence_roi = conf.partial_coherence_roi
            else:
                self.partial_coherence_roi = (16, 16, 16)
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
            if conf.lookup('shrink_wrap_trigger') is not None and conf.lookup('iter_res_sigma_range') is not None:
                # The sigmas are used to find support, if the shrink wrap
                # trigger is not active, it wil not be used
                sigma_range = conf.iter_res_sigma_range
                if len(sigma_range) > 0:
                    first_sigma = sigma_range[0]
                    if len(sigma_range) == 1:
                        last_sigma = self.support_sigma
                    else:
                        last_sigma = sigma_range[1]
                    self.ll_sigmas = [first_sigma + x * (last_sigma - first_sigma) / ll_iter for x in range(ll_iter)]
            self.ll_dets = None
            if conf.lookup('iter_res_det_range') is not None:
                det_range = conf.iter_res_det_range
                if len(det_range) > 0:
                    first_det = det_range[0]
                    if len(det_range) == 1:
                        last_det = 1.0
                    else:
                        last_det = det_range[1]
                    self.ll_dets = [first_det + x * (last_det - first_det) / ll_iter for x in range(ll_iter)]

        return None

