import logging
import os

from hydra.utils import instantiate

import utils

# needed by hydra instantiate
import processor
import interconnect

model_endpoint_repo = None

class ModelEndpointRepo():
    """
    Repository of all hardware configs for dynamic instantiation.
    """
    def __init__(self,
                 model_endpoint_routers_path):
        global model_endpoint_repo
        model_endpoint_repo = self
        self.model_endpoint_router_configs = self.get_model_endpoint_router_configs(model_endpoint_routers_path)

    def get_model_endpoint_router_configs(self, model_endpoint_router_path):
        return utils.read_all_yaml_cfgs(model_endpoint_router_path)

    def get_model_endpoint_router(self, router_name, region, model_name, start_state, scaling_interval, scaling_level, short_term_scaling, **kwargs):
        cfg = self.model_endpoint_router_configs[router_name]
        return instantiate(cfg, region=region, model_name=model_name, start_state=start_state, scaling_interval=scaling_interval, scaling_level=scaling_level,
                           short_term_scaling=short_term_scaling)

get_model_endpoint_router = lambda *args,**kwargs: \
                        model_endpoint_repo.get_model_endpoint_router(*args, **kwargs)
