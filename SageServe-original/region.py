import logging

from abc import ABC, abstractmethod

from model_endpoint_router import ModelEndpointRouter
import model_repo
import orchestrator_repo
import region_repo

from metrics import ApplicationMetrics, ApplicationSLO
from region_router import RegionRouter
from simulator import clock, schedule_event, cancel_event, reschedule_event


class Region():
    """
    A Region is a geographic area that contains multiple clusters of processors.
    """
    def __init__(self,
                 region_name: str,
                 region_id: int,
                 controller,
                 overheads: dict,
                 region_cluster,
                 cache,
                 region_router: RegionRouter,
                 ):
        self.region_name = region_name
        self.region_id = region_id
        self.controller = controller
        self.region_cluster = region_cluster
        self.region_cluster.init_cache(cache.models)
        self.model_endpoint_routers = []
        self.region_router = region_router

        # overheads
        self.overheads = overheads

        # metrics
        self.metrics = ApplicationMetrics()
        self.slo = ApplicationSLO()

    def add_model_endpoint(self, model_endpoint_router: ModelEndpointRouter):
        """
        Application-specific method to add an instance to the application.
        """
        self.model_endpoint_routers.append(model_endpoint_router)
        self.region_router.add_model_endpoint(model_endpoint_router)

    def get_model_endpoint(self, model_name):
        """
        Application-specific method to get an instance from the application.
        """
        for model_endpoint in self.model_endpoint_routers:
            if model_endpoint.model_name == model_name:
                return model_endpoint
        return None

    def get_results(self):
        allocator_results = self.allocator.get_results()
        scheduler_results = self.scheduler.get_results()
        self.scheduler.save_all_request_metrics()
        return allocator_results, scheduler_results
    
    def run(self):
        """
        Run the region. Useful for periodic calls.
        """
        pass

    def __str__(self):
        return f"Region {self.region_name} - {self.region_id} [{self.model_endpoint_routers}]"

    @classmethod
    def from_config(cls, *args, controller, region_cluster, **kwargs):
        region_cfg = args[0]
        region = region_repo.get_region(region_name = region_cfg.region_name,
                                        region_id = region_cfg.region_id,
                                        controller = controller,
                                        region_cluster = region_cluster)
        return region
