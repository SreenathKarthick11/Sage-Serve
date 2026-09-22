import logging

from abc import ABC
from itertools import count

import model_repo

from instance import Instance
from simulator import clock, schedule_event, cancel_event, reschedule_event


class Allocator(ABC):
    """
    Allocator autoscales Application Instances onto Servers.
    It receives/releases Servers from/to Arbiters. 
    """
    def __init__(self,
                 application,
                 arbiter,
                 overheads,
                 instance_overheads,
                 debug=False):
        self.application = application
        self.arbiter = arbiter
        self.overheads = overheads
        self.instance_overheads = instance_overheads
        self.total_instances = count(0)
        self.debug = False
        # self.debug = debug

    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, application):
        self._application = application

    def start_spin_up_instance(self,
                               instance_cfg,
                               processors,
                               parallelism,
                               pre_start=False,
                               tag=None):
        """
        Spin up a new instance of the application on specified processors.
        Assigns a metadata tag to the instance for orchestration.
        # TODO: better way to pass in processors / parallelism
        """
        model_architecture = self.application.model_architecture
        model_size = self.application.model_size
        model = model_repo.get_model(model_architecture=model_architecture,
                                     model_size=model_size,
                                     model_parallelism=parallelism)
        instance = Instance.from_config(instance_cfg=instance_cfg,
                                        instance_id=next(self.total_instances),
                                        application=self.application,
                                        name=processors[0].name,
                                        tag=tag,
                                        model=model,
                                        processors=processors,
                                        overheads=self.instance_overheads,
                                        debug=self.debug)

        # def finish_spin_up():
        #     self.finish_spin_up_instance(instance)
        finish_spin_up = lambda self=self,instance=instance: self.finish_spin_up_instance(instance)
        if pre_start is True:
            finish_spin_up()
        else:
            schedule_event(self.overheads.spin_up, finish_spin_up)

    def finish_spin_up_instance(self, instance):
        """
        Finish spinning up an instance after the spin up delay.
        """
        self.application.add_instance(instance)
        instance.metrics.spin_up_timestamp = clock()
        return instance

    def start_spin_down_instance(self, instance):
        """
        Spin down an instance of the application.
        """
        # def finish_spin_down():
        #     self.finish_spin_down_instance(instance)
        self.application.remove_instance(instance)
        instance.spot = True
        instance.metrics.spin_down_timestamp = clock()
        finish_spin_down = lambda self=self,instance=instance: self.finish_spin_down_instance(instance)
        schedule_event(self.overheads.spin_down, finish_spin_down)
        return instance
    
    def finish_spin_down_instance(self, instance):
        """
        Finish spinning down an instance after the spin down delay.
        """
        region_cluster = instance.application.cluster
        region_cluster.add_spot_instance(instance.application.router.model_name, instance)
        return instance

    # def finish_spin_down_instance(self, instance, processors):
    #     """
    #     Finish spinning down an instance after the spin down delay.
    #     """
    #     instance.memory = 0
    #     pass

    def start_reclaim_spot_instance(self, model_name):
        """
        Reclaim a spot instance.
        """
        region_cluster = self.application.cluster
        instance = region_cluster.remove_spot_instance(model_name)
        instance.spot = False
        # logging.debug("Starting reclaiming " + str(self.overheads.reclaim_spot))
        f = lambda alloc=self, instance=instance: alloc.finish_reclaim_spot_instance(instance)
        schedule_event(self.overheads.reclaim_spot, f)
        return instance
    
    def finish_reclaim_spot_instance(self, instance):
        """
        Finish reclaiming a spot instance.
        """
        # logging.debug("Finishing reclaiming")
        self.application.add_instance(instance)
        instance.application = self.application
        return instance

    def run(self):
        """
        Run the allocator. Useful for periodic calls.
        """
        pass

    def get_results(self):
        results = {}

        instance_names = []
        utilizations = []
        for instance in self.application.instances:
            #assert len(instance.pending_requests) == 0, instance.instance_id
            #assert len(instance.pending_queue) == 0, instance.instance_id
            #assert instance.memory == instance.model.size.total_size, instance.instance_id
            instance.metrics.spin_down_timestamp = clock()
            instance.metrics.interval_time = clock() - instance.metrics.spin_up_timestamp
            # TODO: fix interval=0
            if instance.metrics.interval_time == 0:
                utilization = 0
            else:
                utilization = instance.metrics.busy_time / instance.metrics.interval_time
            instance_names.append(f"{instance.processors[0].name}_{instance.instance_id}")
            utilizations.append(utilization)

        results['instance_names'] = instance_names
        results['utilizations'] = utilizations
        return results


class NoOpAllocator(Allocator):
    """
    No-op Allocator.
    
    Assumes that instances are already allocated to servers using start states.
    """
    pass
