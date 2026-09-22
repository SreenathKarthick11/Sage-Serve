import logging
import random

from collections import defaultdict
from itertools import count

from hydra.utils import instantiate

import cluster_repo
import hardware_repo

from region_router import RegionRouter
from simulator import clock, schedule_event, cancel_event, reschedule_event
from server import Server


class RegionCluster:
    """
    Cluster is a collection of Servers and interconnected Links.
    """
    def __init__(self,
                 region_id,
                #  region,
                 servers,
                 interconnects,
                 power_budget,
                 arbiter = None):
        self.region_id = region_id
        self.region = None
        self.arbiter = arbiter
        self.servers = servers
        self.interconnects = interconnects
        self.power_budget = power_budget
        self.total_power = 0
        self.spot_instances = {} # uses application name
        self.cache = set() # uses model name
        # self.memory = {}
        self.idle_servers = {sku:[] for sku in servers.keys()}
        for sku_name in self.servers:
            for server in self.servers[sku_name]:
                server.cluster = self
                self.total_power += server.power
        self.inflight_commands = []

        # logger for simulated power usage (NOTE: currently unsupported)
        #self.power_logger = utils.file_logger("power")
        #self.power_logger.info("time,server,power")

    def __str__(self):
        return "Cluster:" + str(self.servers)
    
    def init_cache(self, cache_items):
        for model in cache_items:
            self.cache.add(model)

    def set_region(self, region):
        self.region = region
    
    def set_arbiter(self, arbiter):
        self.arbiter = arbiter

    # def add_server(self, server):
    #     self.servers.append(server)

    # def remove_server(self, server):
    #     self.servers.remove(server)

    def models(self):
        models = []
        for server in self.servers:
            models.extend(server.models)
        return models
    
    def add_spot_instance(self, model_type, instance):
        if model_type not in self.spot_instances:
            self.spot_instances[model_type] = []
        self.spot_instances[model_type].append(instance)
    
    def has_spot_instance(self, model_type):
        return model_type in self.spot_instances and len(self.spot_instances[model_type]) > 0
    
    def get_spot_instance_count(self, model_type):
        return len(self.spot_instances.get(model_type, []))
    
    def has_spot_instance_any(self):
        for spot_instance_list in self.spot_instances.values():
            if len(spot_instance_list) > 0:
                return True
        return False
    
    def remove_spot_instance(self, model_type):
        if self.has_spot_instance(model_type):
            instance = self.spot_instances[model_type].pop(0)
        return instance
    
    def free_processors_and_kill_spot(self):
        spot_models = [k for k,v in self.spot_instances.items() if len(v) > 0]
        random_model = random.choice(spot_models)
        instance = self.remove_spot_instance(random_model)
        processors = instance.processors
        del instance
        return processors
    
    def is_cached(self, model_type):
        return model_type in self.cache

    def get_memory(self, model_name):
        """
        Update region endpoint memory usage.
        """
        return self.region.region_router.model_endpoint_routers[model_name].get_memory()

    def get_max_memory(self, model_name):
        """
        Get model endpoint max memory available.
        """
        return self.region.region_router.model_endpoint_routers[model_name].get_max_memory()

    def get_processors(self, affinity=None, count=8):
        preferences = [server_name for server_name in self.servers if affinity in server_name]
        preferences.extend([server_name for server_name in self.servers if server_name not in preferences])
        processors = []
        for preferred_server in preferences:
            for server in self.servers[preferred_server]:
                processors.extend(server.processors)
                if len(processors) >= count:
                    break
            else:
                continue
            break
        return processors

    @property
    def power(self, cached=True, servers=None):
        """
        Returns the total power usage of the cluster.
        Can return the cached value for efficiency.
        TODO: unsupported
        """
        if cached and servers is None:
            return self.total_power
        if servers is None:
            servers = self.servers
        return sum(server.power() for server in servers)

    def update_power(self, power_diff):
        """
        Updates the total power usage of the cluster.
        TODO: unsupported
        """
        self.total_power += power_diff

    def power_telemetry(self, power):
        """
        Logs the power usage of the cluster.
        TODO: currently unsupported; make configurable

        Args:
            power (float): The power usage.
        """
        return
        # time_interval = 60
        # schedule_event(time_interval,
        #                lambda self=self, power=self.total_power: \
        #                    self.power_telemetry(0))

    def run(self):
        """
        Runs servers in the cluster.
        """
        # NOTE: power usage updates not supported
        power = 0
        for sku in self.servers:
            for server in self.servers[sku]:
                server.run()
                power += server.power

    @classmethod
    def from_config(cls, *args, **kwargs):
        # args processing
        cluster_name = args[0].region_cluster
        region_id = args[0].region_id
        # print(args[0])
        cluster_cfg = cluster_repo.get_cluster_cfg(cluster_name)
        # print(cluster_cfg)
        servers_cfg = cluster_cfg.servers
        interconnects_cfg = cluster_cfg.interconnects

        # instantiate servers
        server_id = count()
        servers = defaultdict(list)
        for server_cfg in servers_cfg:
            for n in range(server_cfg.count):
                sku_cfg = hardware_repo.get_sku_config(server_cfg.sku)
                server = Server.from_config(sku_cfg, server_id=next(server_id))
                servers[server_cfg.sku].append(server)

        # instantiate interconnects
        # TODO: add better network topology / configuration support
        interconnects = []
        for interconnect_cfg in interconnects_cfg:
            if interconnect_cfg.topology == "p2p":
                continue
            interconnect = instantiate(interconnect_cfg)
            interconnects.append(interconnect)

        return cls(region_id=region_id,
                #    region=region,
                   servers=servers,
                   interconnects=interconnects,
                   power_budget=cluster_cfg.power_budget)


if __name__ == "__main__":
    pass
