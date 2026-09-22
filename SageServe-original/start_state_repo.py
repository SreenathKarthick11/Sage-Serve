import logging
import os

from hydra.utils import instantiate

import utils

start_state_repo = None

class StartStateRepo():
    """
    Repository of all application configs for dynamic instantiation.
    """
    def __init__(self,
                 start_state_path):
        global start_state_repo
        start_state_repo = self
        self.start_state_configs = self.get_start_state_configs(start_state_path)

    def get_start_state_configs(self, start_states_path):
        return utils.read_all_yaml_cfgs(start_states_path)

    def get_start_state_cfg(self, start_state_name):
        cfg = self.start_state_configs[start_state_name]
        return cfg

get_start_state_cfg = lambda *args,**kwargs: \
                        start_state_repo.get_start_state_cfg(*args, **kwargs)
