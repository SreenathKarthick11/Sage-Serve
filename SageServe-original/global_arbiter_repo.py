import logging
import os

from hydra.utils import instantiate

import utils

# needed by hydra instantiate
import processor
import interconnect

global_arbiter_repo = None

class GlobalArbiterRepo():
    """
    Repository of all global arbiters.
    """
    def __init__(self,
                 global_arbiters_path):
        global global_arbiter_repo
        global_arbiter_repo = self
        self.global_arbiter_configs = self.get_global_arbiter_configs(global_arbiters_path)

    def get_global_arbiter_configs(self, global_arbiters_path):
        return utils.read_all_yaml_cfgs(global_arbiters_path)

    def get_global_arbiter(self, global_arbiter_name, arima_traces, **kwargs):
        print(f"global_arbiter_name: {global_arbiter_name}")
        cfg = self.global_arbiter_configs[global_arbiter_name]
        return instantiate(cfg, arima_traces=arima_traces)

get_global_arbiter = lambda *args,**kwargs: \
            global_arbiter_repo.get_global_arbiter(*args, **kwargs)
