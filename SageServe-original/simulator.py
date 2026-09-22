import heapq
import logging

from collections import defaultdict
import os

import utils
from time import time

from queue import Queue
import psutil

# global simulator that drives the simulation
# bad practice, but it works for now
sim = None

async_count = 0
fed_count = 0
class Event:
    """
    Events are scheduled actions in the simulator.
    """
    def __init__(self, time, action):
        self.time = time
        self.action = action

    def __str__(self):
        return f"Event with time {self.time} and action {self.action}"

    def __lt__(self, other):
        return self.time < other.time


class Simulator:
    """
    A discrete event simulator that schedules and runs Events.
    """
    def __init__(self, end_time):
        global sim
        sim = self
        self.time = 0
        self.end_time = end_time
        self.events = []
        self.deleted_events = []
        logging.info("Simulator initialized")

        # logger for simulator events
        self.logger = utils.file_logger("simulator")
        self.logger.info("time,event")

    def schedule(self, delay, action):
        """
        Schedule an event by specifying delay and an action function.
        """
        # run immediately if delay is 0
        event = Event(self.time + delay, action)
        if delay == 0:
            action()
            return event
        heapq.heappush(self.events, event)
        return event

    def cancel(self, event):
        """
        Cancel an event.
        """
        self.deleted_events.append(event)

    def reschedule(self, event, delay):
        """
        Reschedule an event by cancelling and scheduling it again.
        """
        self.cancel(event)
        return self.schedule(delay, event.action)

    def run(self):
        """
        Run the simulation until the end time.
        """
        while self.events and self.time < self.end_time:
            event = heapq.heappop(self.events)
            if event in self.deleted_events:
                self.deleted_events.remove(event)
                continue
            self.time = event.time
            event.action()
            # self.logger.debug(f"{event.time},{event.action}")


class TraceSimulator(Simulator):
    """
    A discrete event simulator that processes Request arrivals from a Trace.
    """
    def __init__(self,
                 trace,
                 end_time,
                 debug=False):
        super().__init__(end_time)
        self.trace = trace
        self.trace_exhausted = False
        logging.info("TraceSimulator initialized")
        self.batch_prod = {}
        self.opp = {}
        self.debug = False
        # self.debug = debug

        self.last_log_iter = -1

        self.arrival_queue = Queue()

    def schedule_request_arrival(self, delay, action):
        event = Event(self.time + delay, action)
        if delay == 0:
            action()
            return event
        self.arrival_queue.put(event)
        return event

    def feed_async(self):
        """
        Feed async requests from the trace as arrival
        events.
        """
        print("wrong function invoked")
        if len(sim.batch_prod) > 0:
            request = sim.batch_prod.pop(0)
        elif len(sim.opp) > 0:
            request = sim.opp.pop(0)
        else:
            print("Async Queue Empty")
            return
        sim.schedule(0, lambda request=request: sim.global_router.request_arrival(request))

    def run_simulation(self):
        """
        Run the simulation until the end time.
        """
        log_time = 0
        mem_time = 0
        load_trace_time = 0
        mem_write_time = 0
        mem_logs = []
        write_itr = 0
        num_events = 0
        while (self.arrival_queue.qsize() or self.events) and self.time < self.end_time:
            if self.time > load_trace_time or self.arrival_queue.qsize() < (1<<12):
                logging.info(f"Time: {self.time}, load_trace_time: {load_trace_time}, self.arrival_queue.qsize(): {self.arrival_queue.qsize()}")
                if not self.trace_exhausted:
                    self.load_trace()
                    load_trace_time = self.time + (1<<8)
            # event = heapq.heappop(self.events)
            event = None
            if len(self.events) == 0:
                event = self.arrival_queue.get()
            elif self.arrival_queue.qsize() == 0:
                event = heapq.heappop(self.events)
            else:
                event_heap = self.events[0]
                arrival_event = self.arrival_queue.queue[0]
                if event_heap.time < arrival_event.time:
                    event = heapq.heappop(self.events)
                else:
                    event = self.arrival_queue.get()
            if event in self.deleted_events:
                self.deleted_events.remove(event)
                continue
            self.time = event.time
            num_events += 1
            if self.time // 1200 > log_time:
                logging.info(f"Simulation Time: {utils.convert_seconds_to_dd_hh_min_ss(self.time)}s, "\
                             f"Events Queue: {len(self.events)}, Arrival Queue: {self.arrival_queue.qsize()}, "\
                             f"Events executed: {num_events}")
                logging.info(f"Memory utlization: {psutil.virtual_memory().percent}, ")
                log_time = self.time // 1200
            event.action()
            if self.time // 2 > mem_time:
                for reg in self.regions.values():
                    for mer in reg.region_router.model_endpoint_routers.values():
                        for i, app in enumerate(mer.applications):
                            for j, instance in enumerate(app.instances):
                                mem_logs.append(f"{self.time},{reg.region_name}_{mer.model_name}_{i}_{j},{instance.memory-instance.model_memory},{instance.max_memory-instance.model_memory},{instance.model_memory},{len(instance.pending_requests)}\n")
                mem_time = self.time // 2
            if self.time >= mem_write_time + 60*60*1:
                start_time = time()
                logging.info(f"Saving intermediate results at {utils.convert_seconds_to_dd_hh_min_ss(self.time)}")
                os.makedirs("memory", exist_ok=True)
                with open(f"memory/{write_itr}.csv", "w") as f:
                    f.write("time,instance,memory,max_memory,model_memory,pending_requests\n")
                    f.writelines(mem_logs)
                mem_logs = []
                mem_write_time = self.time
                self.save_results_intermediate(write_itr)
                write_itr += 1
                total_time = time() - start_time
                logging.info(f"Intermediate logging took {total_time}s")
                print(async_count, fed_count)
            # self.logger.debug(f"{event.time},{event.action}")
        
        start_time = time()
        # with open(f"memory_{write_itr}.csv", "w") as f:
        #     f.write("time,instance,memory,max_memory,model_memory,pending_requests\n")
        #     for log in mem_logs:
        #         f.write(log)
        total_time = time() - start_time
        logging.info(f"Final result logging took {total_time} seconds")
                
    def load_trace(self):
        requests = self.trace.populate_requests()
        if len(requests) == 0:
            self.trace_exhausted = True
            return
        self.load_trace_batch(requests)
    
    def load_trace_batch(self, requests):
        """
        Load requests from the trace as arrival events.
        """
        for request in requests:
            if request.workload_type == "prod" and request.batch_id == -1:
                self.schedule_request_arrival(request.arrival_timestamp - self.time,
                          lambda self=self,request=request: self.global_router.request_arrival(request))
            elif request.workload_type == "prod" and request.batch_id != -1:
                if request.model_type not in self.batch_prod:
                    self.batch_prod[request.model_type] = {}
                if request.region_priority[0] not in self.batch_prod[request.model_type]:
                    self.batch_prod[request.model_type][request.region_priority[0]] = []
                self.batch_prod[request.model_type][request.region_priority[0]].append(request)
            else:
                if request.model_type not in self.opp:
                    self.opp[request.model_type] = {}
                if request.region_priority[0] not in self.opp[request.model_type]:
                    self.opp[request.model_type][request.region_priority[0]] = []
                self.opp[request.model_type][request.region_priority[0]].append(request)
        # TODO: assuming the requests are sorted already
        # for model_type in self.batch_prod:
        #     for region_id in self.batch_prod[model_type]:
        #         self.batch_prod[model_type][region_id].sort(key=lambda x: x.arrival_timestamp)
        # for model_type in self.opp:
        #     for region_id in self.opp[model_type]:
        #         self.opp[model_type][region_id].sort(key=lambda x: x.arrival_timestamp)
  
    def add_controller(self, controller):
        self.controller = controller
    def add_global_router(self, global_router):
        self.global_router = global_router
    def add_region_clusters(self, region_clusters): 
        self.region_clusters = region_clusters
    def add_regions(self, regions):
        self.regions = regions
    def add_model_endpoint_routers(self, model_endpoint_routers):
        self.model_endpoint_routers = model_endpoint_routers
    def add_applications(self, applications):
        self.applications = applications

    def run(self):
        # start simulation by scheduling a cluster run
        self.schedule(0, self.controller.run)
        self.schedule(0, self.global_router.run)
        for model_endpoint_router in self.model_endpoint_routers:
            self.schedule(0, model_endpoint_router.run)
        for region in self.regions.values():
            self.schedule(0, region.run)
        # self.schedule(0, self.arbiter.run)


        # run simulation
        # super().run()
        self.run_simulation()
        self.logger.info(f"{self.time},end")
        logging.info(f"TraceSimulator completed at {self.time}")

        # self.save_results()

    def save_results_intermediate(self, write_itr, detailed=True):
        self.global_router.save_results_intermediate(write_itr)
        for region in self.regions.values():
            region.region_router.save_results_intermediate(write_itr)
            region.region_cluster.arbiter.save_results_intermediate(write_itr)

        sched_results = {}
        for application in self.applications:
            application_identifier = f"{application.region.region_name}_{application.router.model_name}_{application.application_id}"
            scheduler_results = application.get_results_intermediate(write_itr)
            sched_results[application_identifier] = scheduler_results
        
        summary_results = defaultdict(list)
        for application_id, results_dict in sched_results.items():
            summary_results["application_id"].append(application_id)
            for key, values in results_dict.items():
                summary = utils.get_statistics(values)
                for metric, value in summary.items():
                    summary_results[f"{key}_{metric}"].append(value)

        utils.save_dict_as_csv(summary_results, f"summary/{write_itr}.csv")
        self.last_log_iter = write_itr

        if detailed:
            # create a dataframe of all requests, save as csv
            for application_id, result in sched_results.items():
                utils.save_dict_as_csv(result, f"detailed/{application_id}/{write_itr}.csv")

    def save_results(self, detailed=True):
        """
        Save results at the end of the simulation.
        """
        self.global_router.save_results()
        for region in self.regions.values():
            region.region_router.save_results()
            region.region_cluster.arbiter.save_results()

        sched_results = {}
        alloc_results = {}
        for application in self.applications:
            application_identifier = f"{application.region.region_name}_{application.router.model_name}_{application.application_id}"
            allocator_results, scheduler_results = application.get_results()
            alloc_results[application_identifier] = allocator_results
            sched_results[application_identifier] = scheduler_results

        # summary sched results
        summary_results = defaultdict(list)
        for application_id, results_dict in sched_results.items():
            summary_results["application_id"].append(application_id)
            for key, values in results_dict.items():
                summary = utils.get_statistics(values)
                # merge summary into summary_results
                for metric, value in summary.items():
                    summary_results[f"{key}_{metric}"].append(value)

        # save summary results
        utils.save_dict_as_csv(summary_results, f"summary/{self.last_log_iter + 1}.csv")

        if detailed:
            # create a dataframe of all requests, save as csv
            for application_id, result in sched_results.items():
                utils.save_dict_as_csv(result, f"detailed/{application_id}/{self.last_log_iter + 1}.csv")
            for application_id, result in alloc_results.items():
                utils.save_dict_as_csv(result, f"detailed/{application_id}_alloc/{self.last_log_iter + 1}.csv")


# Convenience functions for simulator object

def clock():
    """
    Return the current time of the simulator.
    """
    return sim.time

def schedule_event(*args):
    """
    Schedule an event in the simulator at desired delay.
    """
    return sim.schedule(*args)

def cancel_event(*args):
    """
    Cancel existing event in the simulator.
    """
    return sim.cancel(*args)

def reschedule_event(*args):
    """
    Reschedule existing event in the simulator.
    Equivalent to cancelling and scheduling a new event.
    """
    return sim.reschedule(*args)

def feed_async(region, model_name, feed_async_granularity):
    mem_util = region.region_cluster.get_memory(model_name) / region.region_cluster.get_max_memory(model_name)
    if mem_util > 0.6:
        return
    if mem_util > 0.5:
        for _ in range(feed_async_granularity):
            feed_async_helper(region, model_name)
    for _ in range(feed_async_granularity):
        feed_async_helper(region, model_name)

def feed_async_helper(region, model_name):
    """
    Feed n async requests from the trace as arrival events.
    """
    global async_count
    
    # mem_util = region.region_cluster.get_memory(model_name) / region.region_cluster.get_max_memory(model_name)
    # print(mem_util)
    # num_async = 0
    # if mem_util > 0.6:
    #     # print(clock(), "Memory usage high, not feeding", region.region_name, model_name, mem_util)
    #     num_async = 0
    # elif mem_util > 0.3:
    #     num_async = 1
    # else:
    #     num_async = 2
    # # print(region.region_name, model_name, sim.opp.keys())
    # for _ in range(num_async):
    if model_name in sim.batch_prod and region.region_id in sim.batch_prod[model_name]\
            and len(sim.batch_prod[model_name][region.region_id]) > 0:
        request = sim.batch_prod[model_name][region.region_id].pop(0)
    elif model_name in sim.opp and region.region_id in sim.opp[model_name]\
            and len(sim.opp[model_name][region.region_id]) > 0:
        request = sim.opp[model_name][region.region_id].pop(0)
    else:
        # print("Async Queue Empty", region.region_name, model_name)
        # for m in sim.opp.keys():
        #     print([(m, r, len(c)) for r, c in sim.opp[m].items()])
        return
    async_count += 1
    # print(clock(), "Feeding async request", region.region_name, model_name, request.request_id)
    sim.schedule(0, lambda request=request: sim.global_router.request_arrival(request))

def inc_fed_count():
    global fed_count
    fed_count += 1
