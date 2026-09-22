import logging

from collections import defaultdict
from itertools import count

from hydra.utils import instantiate

import hardware_repo

from simulator import clock, schedule_event, cancel_event, reschedule_event
from region_cluster import RegionCluster


class Controller:
    """
    Controller is a collection of Regions.
    """
    def __init__(self,
                 power_budget,
                 global_router=None,
                 regions=[]):
        self.regions = regions
        # self.interconnects = interconnects
        self.power_budget = power_budget
        self.global_router = global_router
        self.total_power = 0
        self.memory = {}
        for region_name in self.regions:
            for model_endpoint in self.regions[region_name]:
                model_endpoint.controller = self
                self.total_power += model_endpoint.power
        self.inflight_commands = []

        # logger for simulated power usage (NOTE: currently unsupported)
        #self.power_logger = utils.file_logger("power")
        #self.power_logger.info("time,server,power")

    def __str__(self):
        return "Controller:" + str(self.regions)

    def set_global_router(self, global_router):
        self.global_router = global_router
    
    def add_region(self, region):
        self.regions.append(region)
    
    def remove_region(self, region):
        self.regions.remove(region)

    def models(self):
        models = []
        for server in self.servers:
            models.extend(server.models)
        return models

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

    def update_memory(self, region_name, model_name, memory):
        """
        Update global memory usage.
        """
        if region_name not in self.memory:
            self.memory[region_name] = {}
        if model_name not in self.memory[region_name]:
            self.memory[region_name][model_name] = 0
        self.memory[region_name][model_name] += memory

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
        # TODO
        pass
        # power = 0
        # for sku in self.servers:
        #     for server in self.servers[sku]:
        #         server.run()
        #         power += server.power

    @classmethod
    def from_config(cls, *args, **kwargs):
        # args processing
        controller_cfg = args[0]
        # regions_cfg = controller_cfg.regions
        # interconnects_cfg = cluster_cfg.interconnects

        # instantiate servers
        # server_id = count()
        # regions = []
        # for region_cfg in regions_cfg:
        #     print(region_cfg)
        #     region = Region.from_config(region_cfg)
        #     regions.append(region)
            # for n in range(server_cfg.count):
            #     sku_cfg = hardware_repo.get_sku_config(server_cfg.sku)
            #     server = Server.from_config(sku_cfg, server_id=next(server_id))
            #     servers[server_cfg.sku].append(server)

        # instantiate interconnects
        # TODO: add better network topology / configuration support
        # interconnects = []
        # for interconnect_cfg in interconnects_cfg:
        #     if interconnect_cfg.topology == "p2p":
        #         continue
        #     interconnect = instantiate(interconnect_cfg)
        #     interconnects.append(interconnect)

        return cls(power_budget=controller_cfg.power_budget)


if __name__ == "__main__":
    pass
