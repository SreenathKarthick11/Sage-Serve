import logging
import os

from hydra.utils import instantiate

import utils

# needed by hydra instantiate
import processor
import interconnect

application_repo = None

class ApplicationRepo():
    """
    Repository of all application configs for dynamic instantiation.
    """
    def __init__(self,
                 applications_path):
        global application_repo
        application_repo = self
        self.application_configs = self.get_application_configs(applications_path)

    def get_application_configs(self, applications_path):
        return utils.read_all_yaml_cfgs(applications_path)

    def get_application_cfg(self, application_name):
        cfg = self.application_configs[application_name]
        return cfg

get_application_cfg = lambda *args,**kwargs: \
                        application_repo.get_application_cfg(*args, **kwargs)
