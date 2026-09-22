"""
Utility functions for initializing the simulation environment.
"""

import logging
import os

from hydra.utils import instantiate
from hydra.utils import get_original_cwd

# from application import Application
# from cluster import Cluster
from application import Application
from application_repo import ApplicationRepo
import application_repo
from arbiter import Arbiter
import arbiter_repo
from controller import Controller
from hardware_repo import HardwareRepo
from model_endpoint_repo import ModelEndpointRepo
from arbiter_repo import ArbiterRepo
from  global_arbiter_repo import GlobalArbiterRepo
import model_endpoint_repo
from model_repo import ModelRepo
from cluster_repo import ClusterRepo
# from orchestrator_repo import OrchestratorRepo
# from start_state import load_start_state
from trace import Trace

from orchestrator_repo import OrchestratorRepo
from region import Region
from region_cluster import RegionCluster
from region_repo import RegionRepo
from start_state import load_start_state
from start_state_repo import StartStateRepo
import start_state_repo


def init_trace(cfg):
    trace_path = os.path.join(get_original_cwd(), cfg.trace.path)
    trace = Trace.from_csv(trace_path, cfg.siloed)
    return trace

def init_cluster_repo(cfg):
    clusters_path = os.path.join(get_original_cwd(), cfg.cluster_repo.clusters)
    cluster_repo = ClusterRepo(clusters_path)
    return cluster_repo

def init_hardware_repo(cfg):
    processors_path = os.path.join(get_original_cwd(),
                                   cfg.hardware_repo.processors)
    interconnects_path = os.path.join(get_original_cwd(),
                                      cfg.hardware_repo.interconnects)
    skus_path = os.path.join(get_original_cwd(),
                             cfg.hardware_repo.skus)
    hardware_repo = HardwareRepo(processors_path,
                                 interconnects_path,
                                 skus_path)
    return hardware_repo


def init_model_repo(cfg):
    model_architectures_path = os.path.join(get_original_cwd(),
                                            cfg.model_repo.architectures)
    model_sizes_path = os.path.join(get_original_cwd(),
                                    cfg.model_repo.sizes)
    model_repo = ModelRepo(model_architectures_path, model_sizes_path)
    return model_repo


def init_orchestrator_repo(cfg):
    allocators_path = os.path.join(get_original_cwd(),
                                   cfg.orchestrator_repo.allocators)
    schedulers_path = os.path.join(get_original_cwd(),
                                   cfg.orchestrator_repo.schedulers)
    orchestrator_repo = OrchestratorRepo(allocators_path, schedulers_path)
    return orchestrator_repo

def init_region_repo(cfg):
    regions_path = os.path.join(get_original_cwd(),
                                cfg.region_repo.regions)
    region_routers_path = os.path.join(get_original_cwd(),
                                       cfg.region_repo.region_routers)
    caches_path = os.path.join(get_original_cwd(),
                                       cfg.region_repo.caches)
    region_repo = RegionRepo(regions_path, region_routers_path, caches_path)
    return region_repo

def init_arbiter_repo(cfg):
    arbiters_path = os.path.join(get_original_cwd(),
                                cfg.arbiter_repo.arbiters)
    arbiter_repo = ArbiterRepo(arbiters_path)
    return arbiter_repo


def init_model_endpoint_repo(cfg):
    model_endpoints_path = os.path.join(get_original_cwd(),
                                        cfg.model_endpoint_repo.model_endpoint_routers)
    model_endpoint_repo = ModelEndpointRepo(model_endpoints_path)
    return model_endpoint_repo

def init_application_repo(cfg):
    applications_path = os.path.join(get_original_cwd(),
                                    cfg.application_repo.applications)
    application_repo = ApplicationRepo(applications_path)
    return application_repo

def init_start_state_repo(cfg):
    start_states_path = os.path.join(get_original_cwd(),
                                    cfg.start_state_repo.start_states)
    start_state_repo = StartStateRepo(start_states_path)
    return start_state_repo

def init_performance_model(cfg):
    performance_model = instantiate(cfg.performance_model)
    return performance_model


def init_power_model(cfg):
    power_model = instantiate(cfg.power_model)
    return power_model


def init_controller(cfg):
    cluster = Controller.from_config(cfg.controller)
    return cluster

def init_global_router(cfg, controller):
    global_router = instantiate(cfg.global_router, 
                                controller=controller, 
                                long_term_scaling=cfg.long_term_scaling)
    return global_router

def init_global_arbiter(cfg):
    global_arbiter = instantiate(cfg.global_arbiter)
    return global_arbiter

def init_controller(cfg):
    controller = Controller.from_config(cfg.controller)
    return controller

def init_regions(cfg, controller):
    model_endpoint_routers = []
    applications = []
    regions = {}
    region_clusters = {}
    # print(cfg.controller.regions)
    for region_cfg in cfg.controller.regions:
        # print(region_cfg)
        region_cluster = RegionCluster.from_config(region_cfg)
        region:Region = Region.from_config(region_cfg,
                                    controller=controller, region_cluster=region_cluster)
        region_cluster.set_region(region)
        arbiter:Arbiter = arbiter_repo.get_arbiter(region_cfg.arbiter, cluster=region_cluster, scaling_threshhold=cfg.arima_checking_scaling_threshhold)
        region_cluster.set_arbiter(arbiter)
        for model_endpoint_cfg in region_cfg.model_endpoints:
            # model_cluster = ModelCluster.from_config(model_cluster_cfg, region=region)
            model_endpoint_router = model_endpoint_repo.get_model_endpoint_router(model_endpoint_cfg.model_endpoint_router,
                                                                                  region, model_endpoint_cfg.model_name,
                                                                                  model_endpoint_cfg.start_state,
                                                                                  cfg.scaling_interval,
                                                                                  cfg.scaling_level,
                                                                                  cfg.short_term_scaling)
            if region_cfg.arbiter == "chiron_arbiter":
                model_endpoint_router.set_prompt_time(model_endpoint_cfg.prompt_time)
                model_endpoint_router.set_decode_throughput(model_endpoint_cfg.decode_throughput)
            region.add_model_endpoint(model_endpoint_router)
            total_instances = model_endpoint_cfg.instance_count
            if total_instances % 3 == 1:
                instances = [3 for _ in range(total_instances//3 - 1)]
                instances.extend([2, 2])
            elif total_instances % 3 == 2:
                instances = [3 for _ in range(total_instances//3)]
                instances.append(2)
            else:
                instances = [3 for _ in range(total_instances//3)]
            assert sum(instances) == total_instances
            for idx, instance_count in enumerate(instances):
                application = Application.from_config(application_repo.get_application_cfg(model_endpoint_cfg.model_name),
                                            application_id=idx,
                                            cluster=region_cluster,
                                            region=region,
                                            router=model_endpoint_router,
                                            arbiter=None,
                                            feed_async=cfg.feed_async,
                                            feed_async_granularity=cfg.feed_async_granularity)
                start_state_cfg = start_state_repo.get_start_state_cfg(model_endpoint_cfg.start_state)
                load_start_state(start_state_cfg, cluster=region_cluster, application=application, count=instance_count)
                model_endpoint_router.add_application(application)
                applications.append(application)
            # for application_cfg in model_endpoint_cfg.deployments:
            #     # print(application_cfg)
            #     application = Application.from_config(application_repo.get_application_cfg(application_cfg.application_name),
            #                                 application_id=application_cfg.application_id,
            #                                 cluster=region_cluster,
            #                                 router=model_endpoint_router,
            #                                 arbiter=None)
            #     start_state_cfg = start_state_repo.get_start_state_cfg(application_cfg.start_state)
            #     load_start_state(start_state_cfg, cluster=region_cluster, application=application)
            #     model_endpoint_router.add_application(application)
            #     applications[application_cfg.application_id] = application
            model_endpoint_routers.append(model_endpoint_router)
        regions[region_cfg.region_id] = region
        region_clusters[region_cfg.region_id] = region_cluster
    return regions, region_clusters, model_endpoint_routers, applications

def init_router(cfg, cluster):
    router = instantiate(cfg.router, cluster=cluster)
    return router


def init_arbiter(cfg, cluster):
    arbiter = instantiate(cfg.arbiter, cluster=cluster)
    return arbiter


def init_application(cfg, cluster, router, arbiter):
    application = Application.from_config(cfg,
                                            cluster=cluster,
                                            router=router,
                                            arbiter=arbiter)
    return application


def init_start_state(cfg, **kwargs):
    load_start_state(cfg.start_state, **kwargs)


if __name__ == "__main__":
    pass
