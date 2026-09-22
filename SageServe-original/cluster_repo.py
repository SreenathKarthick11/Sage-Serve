import logging
import os

from hydra.utils import instantiate

import utils

# needed by hydra instantiate
import processor
import interconnect

cluster_repo = None

class ClusterRepo():
    """
    Repository of all hardware configs for dynamic instantiation.
    """
    def __init__(self,
                 clusters_path):
        global cluster_repo
        cluster_repo = self
        self.cluster_configs = self.get_cluster_configs(clusters_path)

    def get_cluster_configs(self, clusters_path):
        return utils.read_all_yaml_cfgs(clusters_path)


    def get_cluster_cfg(self, cluster_name):
        cfg = self.cluster_configs[cluster_name]
        return cfg

get_cluster_cfg = lambda *args,**kwargs: \
                        cluster_repo.get_cluster_cfg(*args, **kwargs)
