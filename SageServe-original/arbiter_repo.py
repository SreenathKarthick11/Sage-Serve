import logging
import os

from hydra.utils import instantiate

import utils

# needed by hydra instantiate
import processor
import interconnect

arbiter_repo = None

class ArbiterRepo():
    """
    Repository of all arbiters for different clusters.
    """
    def __init__(self,
                 arbiters_path):
        global arbiter_repo
        arbiter_repo = self
        self.arbiter_configs = self.get_arbiter_configs(arbiters_path)

    def get_arbiter_configs(self, arbiters_path):
        return utils.read_all_yaml_cfgs(arbiters_path)


    def get_arbiter(self, arbiter_name, cluster, **kwargs):
        cfg = self.arbiter_configs[arbiter_name]
        return instantiate(cfg, cluster=cluster, **kwargs)

get_arbiter = lambda *args,**kwargs: \
            arbiter_repo.get_arbiter(*args, **kwargs)
