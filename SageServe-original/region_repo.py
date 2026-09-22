import logging
import os

from hydra.utils import instantiate

# from region import Region
import utils

# needed by hydra instantiate
# import allocator
# import scheduler


region_repo = None


class RegionRepo():
    """
    Repository of all region configs.
    """
    def __init__(self,
                 regions_path,
                 region_routers_path,
                 caches_path):
        global region_repo
        region_repo = self
        self.region_configs = self.get_region_configs(regions_path)
        self.region_router_configs = self.get_region_router_configs(region_routers_path)
        self.caches_config = self.get_caches_configs(caches_path)

    def get_region_configs(self, regions_path):
        return utils.read_all_yaml_cfgs(regions_path)

    def get_region_router_configs(self, region_routers_path):
        return utils.read_all_yaml_cfgs(region_routers_path)
    
    def get_caches_configs(self, caches_path):
        return utils.read_all_yaml_cfgs(caches_path)

    def get_cache_config(self, cache_name):
        return self.caches_config[cache_name]

    def get_region(self, region_name, region_id, controller, region_cluster, **kwargs):
        cfg = self.region_configs[region_name]
        # print(cfg)
        cache = get_cache_config(cfg.cache)
        region_router = get_region_router(router_name=cfg.region_router)
        region = instantiate(cfg,
                           region_id = region_id,
                           controller = controller,
                           region_router=region_router,
                           cache=cache,
                           region_cluster=region_cluster)
        region_router.set_region(region)
        return region

    def get_region_router(self, router_name, **kwargs):
        cfg = self.region_router_configs[router_name]
        return instantiate(cfg)


get_region = lambda *args,**kwargs: \
            region_repo.get_region(*args, **kwargs)
get_region_router = lambda *args,**kwargs: \
            region_repo.get_region_router(*args, **kwargs)
get_cache_config = lambda *args,**kwargs: \
            region_repo.get_cache_config(*args, **kwargs)
