import logging

from abc import ABC, abstractmethod

import model_repo
import orchestrator_repo

from metrics import ApplicationMetrics, ApplicationSLO
from simulator import clock, schedule_event, cancel_event, reschedule_event


class Application():
    """
    An Application is the endpoint that a Request targets.
    Applications can have any number of Instances which all serve the same model.
    Requests are scheduled to Instances by the Scheduler.
    Application Instances can be autoscaled by the Allocator.
    """
    def __init__(self,
                 application_id,
                 model_architecture,
                 model_size,
                 cluster,
                 region,
                 router,
                 arbiter,
                 overheads,
                 feed_async=True,
                 feed_async_granularity=1,
                 processor_affinity=None,
                 processor_count=8,
                 scheduler=None,
                 allocator=None,
                 instances=None):
        self.application_id = application_id

        # hardware
        self.processors = []

        # model
        self.model_architecture = model_architecture
        self.model_size = model_size

        # orchestration
        if instances is None:
            self.instances = []
        self.cluster = cluster
        self.region = region
        self.scheduler = scheduler
        self.allocator = allocator
        self.router = router
        self.arbiter = arbiter
        self.processor_affinity = processor_affinity
        self.processor_count = processor_count
        self.feed_async = feed_async
        self.feed_async_granularity = feed_async_granularity

        # overheads
        self.overheads = overheads

        # metrics
        self.metrics = ApplicationMetrics()
        self.slo = ApplicationSLO()

    def add_instance(self, instance):
        """
        Application-specific method to add an instance to the application.
        """
        self.instances.append(instance)
        self.scheduler.add_instance(instance)
    
    def remove_instance(self, instance):
        """
        Application-specific method to remove an instance from the application.
        """
        self.instances.remove(instance)
        self.scheduler.remove_instance(instance)
    
    def get_memory(self):
        """
        Get application memory usage.
        """
        memory = 0
        for instance in self.instances:
            memory += instance.memory - instance.model_memory
        return memory
    
    def get_max_memory(self):
        """
        Get application max memory available.
        """
        max_memory = 0
        for instance in self.instances:
            max_memory += instance.max_memory - instance.model_memory
        return max_memory
    
    def get_max_instance_util(self):
        return max([instance.memory / instance.max_memory for instance in self.instances])

    def get_results_intermediate(self, write_itr):
        scheduler_results = self.scheduler.get_results()
        self.scheduler.save_all_request_metrics_intermediate(write_itr)
        return scheduler_results

    def get_results(self):
        allocator_results = self.allocator.get_results()
        scheduler_results = self.scheduler.get_results()
        self.scheduler.save_all_request_metrics()
        return allocator_results, scheduler_results
    
    def __str__(self) -> str:
        return f"Application {self.application_id} [{len(self.instances)}]"

    @classmethod
    def from_config(cls, *args, application_id, cluster, region, arbiter, router, feed_async, feed_async_granularity, **kwargs):
        # parse args
        application_cfg = args[0]

        # get model
        model_architecture_name = application_cfg.model_architecture
        model_size_name = application_cfg.model_size
        model_architecture = model_repo.get_model_architecture(model_architecture_name)
        model_size = model_repo.get_model_size(model_size_name)

        # get orchestrators
        allocator_name = application_cfg.allocator
        scheduler_name = application_cfg.scheduler
        processor_affinity = application_cfg.processor_affinity
        processor_count = application_cfg.processor_count
        application = cls(application_id=application_id,
                          model_architecture=model_architecture,
                          model_size=model_size,
                          cluster=cluster,
                          region=region,
                          router=router,
                          arbiter=arbiter,
                          processor_affinity=processor_affinity,
                          processor_count=processor_count,
                          overheads=application_cfg.overheads,
                          feed_async=feed_async,
                          feed_async_granularity=feed_async_granularity)
        allocator = orchestrator_repo.get_allocator(allocator_name,
                                                    arbiter=arbiter,
                                                    application=application,
                                                    debug=application_cfg.debug)
        scheduler = orchestrator_repo.get_scheduler(scheduler_name,
                                                    router=router,
                                                    application=application,
                                                    debug=application_cfg.debug)
        application.scheduler = scheduler
        application.allocator = allocator
        return application
