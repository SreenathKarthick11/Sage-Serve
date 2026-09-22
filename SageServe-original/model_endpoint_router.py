import logging

from abc import ABC, abstractmethod
import random

from request import Request
from simulator import clock, schedule_event, cancel_event, reschedule_event

from time import time

class ModelEndpointRouter(ABC):
    """
    Router routes Requests to Application Schedulers.
    """
    def __init__(self,
                 region,
                 model_name,
                 start_state,
                 scaling_interval,
                 scaling_level,
                 short_term_scaling,
                 overheads):
        self.region = region
        self.model_name = model_name
        self.overheads = overheads
        self.start_state = start_state
        self.applications = []
        self.schedulers = {}

        # request queues
        # self.pending_queue = []
        # self.executing_queue = []
        # self.completed_queue = []
        self.pending_requests = 0
        self.pending_tokens = 0

        self.scaling_interval = scaling_interval
        self.scaling_level = scaling_level
        self.last_scale_timestamp = -1e9

        self.short_term_scaling = short_term_scaling

    def add_application(self, application):
        self.applications.append(application)
        self.schedulers[application.application_id] = application.scheduler

    def remove_application(self, application):
        self.applications.remove(application)
        self.schedulers.pop(application.application_id)

    def run(self):
        pass

    @property
    def total_instances(self):
        if len(self.schedulers.values()) == 0:
            return 0
        return sum([scheduler.instance_count for scheduler in self.schedulers.values()])

    @abstractmethod
    def route(self, *args):
        """
        Main routing logic
        """
        raise NotImplementedError

    def request_arrival(self, request:Request):
        request.arrive_at_model_endpoint_router()
        # self.pending_queue.append(request)
        self.pending_requests += 1
        self.pending_tokens += request.token_size
        self.route_request(request)

    def request_completion(self, request:Request):
        request.complete_at_model_endpoint_router()
        # self.executing_queue.remove(request)
        self.pending_requests -= 1
        self.pending_tokens -= request.token_size
        # self.completed_queue.append(request)
        self.region.region_router.request_completion(request)

    def route_request(self, request):
        self.route(request)
        # self.pending_queue.remove(request)
        # self.executing_queue.append(request)

    def get_memory(self):
        memory = 0
        for application in self.applications:
            memory += application.get_memory()
        return memory

    def get_max_memory(self):
        """
        Get model endpoint max memory available.
        """
        max_memory = 0
        for application in self.applications:
            max_memory += application.get_max_memory()
        return max_memory
    
    def get_max_instance_util(self):
        return max([app.get_max_instance_util() for app in self.applications])

    def save_results(self):
        pass

    def __str__(self):
        return f"ModelEndpointRouter {self.model_name} - {self.region.region_name}"


class RoundRobin(ModelEndpointRouter):
    """
    NOTE: This is JSQ
    TODO: Rename
    """
    def __init__(self, region, model_name, start_state, scaling_interval, scaling_level, short_term_scaling, overheads):
        super().__init__(region, model_name, start_state, scaling_interval, scaling_level, short_term_scaling, overheads)
        self.idx = -1
        self.current_request = None
    """
    Forwards request to the appropriate scheduler without any overheads.
    """
    def route(self, request):
        # start_time = time()
        if self.short_term_scaling and clock() > self.last_scale_timestamp + self.scaling_interval:
            if self.short_term_scaling:
                self.current_request = request
                self.region.region_cluster.arbiter.scale(self)
                self.last_scale_timestamp = clock()

        # app_id = min(self.applications, key=lambda a: a.get_memory() / a.get_max_memory()).application_id
        app_id = min(self.applications, key=lambda a: min([i.pending_tokens for i in a.instances])).application_id
        scheduler = self.schedulers[list(self.schedulers.keys())[app_id]]


        f = lambda scheduler=scheduler,request=request: \
            scheduler.request_arrival(request)
        schedule_event(self.overheads.routing_delay, f)

class ChironRoundRobin(RoundRobin):
    """
    Chiron model endpoint router.
    """
    
    def __init__(self, region, model_name, start_state, scaling_interval, scaling_level, short_term_scaling, overheads):
        super().__init__(region, model_name, start_state, scaling_interval, scaling_level, short_term_scaling, overheads)
        self.prompt_time = 10
        self.decode_throughput = 100
    def set_prompt_time(self, prompt_time):
        """
        Set the prompt time for the model endpoint router.
        """
        self.prompt_time = prompt_time
    def set_decode_throughput(self, decode_throughput):
        """
        Set the decode throughput for the model endpoint router.
        """
        self.decode_throughput = decode_throughput
