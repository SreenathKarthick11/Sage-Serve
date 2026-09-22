import logging

from abc import ABC, abstractmethod

from request import Request
from simulator import clock, schedule_event, cancel_event, reschedule_event
import utils

from time import time

class RegionRouter(ABC):
    """
    Router routes Requests to Model Endpoint Schedulers.
    """
    def __init__(self,
                 overheads,
                 region=None):
        self.region = region
        self.overheads = overheads
        self.model_endpoint_routers = {}

        # request queues
        self.pending_queue = []
        self.results = {
            "request_id": [],
            "model": [],
            "workload_type": [],
            "arrival_timestamp": [],
            "completion_timestamp": [],
            "global_router_queue_time": [],
            "region_router_queue_time": [],
            "model_endpoint_router_queue_time": [],
            "application_scheduler_queue_time": [],
            "service_time": [],
            "application_scheduler_response_time": [],
            "model_endpoint_router_response_time": [],
            "region_router_response_time": [],
            "global_router_response_time": [],
            "queue_time": []
        }

        self.last_log_iter = -1
    
    def set_region(self, region):
        self.region = region

    def add_model_endpoint(self, model_endpoint_router):
        self.model_endpoint_routers[model_endpoint_router.model_name] = model_endpoint_router

    def run(self):
        pass

    @abstractmethod
    def route(self, *args):
        """
        Main routing logic
        """
        raise NotImplementedError

    def request_arrival(self, request: Request):
        request.arrive_at_region_router()
        self.route_request(request)

    def request_completion(self, request: Request):
        request.complete_at_region_router()
        self.results['request_id'].append(request.request_id)
        self.results['model'].append(request.model_type)
        self.results['workload_type'].append(request.workload_type)
        self.results['arrival_timestamp'].append(request.metrics.global_router_arrival_timestamp)
        self.results['completion_timestamp'].append(request.metrics.global_router_completion_timestamp)
        self.results['global_router_queue_time'].append(request.metrics.global_router_queue_time)
        self.results['region_router_queue_time'].append(request.metrics.region_router_queue_time)
        self.results['model_endpoint_router_queue_time'].append(request.metrics.model_endpoint_router_queue_time)
        self.results['application_scheduler_queue_time'].append(request.metrics.application_scheduler_queue_time)
        self.results['service_time'].append(request.metrics.service_time)
        self.results['application_scheduler_response_time'].append(request.metrics.application_scheduler_response_time)
        self.results['model_endpoint_router_response_time'].append(request.metrics.model_endpoint_router_response_time)
        self.results['region_router_response_time'].append(request.metrics.region_router_response_time)
        self.results['global_router_response_time'].append(request.metrics.global_router_response_time)
        self.results['queue_time'].append(request.metrics.queue_time)
        self.region.controller.global_router.request_completion(request)

    def route_request(self, request):
        self.route(request)

    def save_results_intermediate(self, write_itr):
        utils.save_dict_as_csv(self.results, f"region_routers/{self.region.region_name}/{write_itr}.csv")
        self.results = {
            "request_id": [],
            "model": [],
            "workload_type": [],
            "arrival_timestamp": [],
            "completion_timestamp": [],
            "global_router_queue_time": [],
            "region_router_queue_time": [],
            "model_endpoint_router_queue_time": [],
            "application_scheduler_queue_time": [],
            "service_time": [],
            "application_scheduler_response_time": [],
            "model_endpoint_router_response_time": [],
            "region_router_response_time": [],
            "global_router_response_time": [],
            "queue_time": []
        }
        self.last_log_iter = write_itr

    def save_results(self):
        utils.save_dict_as_csv(self.results, f"region_routers/{self.region.region_name}/{self.last_log_iter + 1}.csv")


class StaticRouter(RegionRouter):
    """
    Forwards request to the appropriate model endpoint without any overheads.
    """
    def route(self, request: Request):
        assert request.model_type in self.model_endpoint_routers
        model_router = self.model_endpoint_routers[request.model_type]
        f = lambda model_router=model_router,request=request: \
            model_router.request_arrival(request)
        schedule_event(self.overheads.routing_delay, f)

class ChironRegionRouter(RegionRouter):
    """
    Forwards request to the appropriate model endpoint with overheads.
    """
    def route(self, request: Request):
        assert request.model_type in self.model_endpoint_routers
        model_router = self.model_endpoint_routers[request.model_type]
        if model_router.get_memory() / model_router.get_max_memory() > 0.7:
            # print("routing to mixed", request.model_type)
            model_router = self.model_endpoint_routers[request.model_type[0]] # remove _d or _p suffix
        f = lambda model_router=model_router,request=request: \
            model_router.request_arrival(request)
        schedule_event(self.overheads.routing_delay, f)
