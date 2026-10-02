import logging
from abc import ABC
from typing import List, Tuple
import pandas as pd
import os

from simulator import clock, schedule_event
import utils
from long_term_allocation import MilpLongTermAllocation


class GlobalArbiter(ABC):
    """
    Global Arbiter allocates Models to Regions from the global router.
    """

    def __init__(self, long_term_scaling_interval, max_time):
        self.regions = 3
        self.models = 0
        self.model_to_idx_mapping = {}
        self.model_list = []
        self.long_term_scaling_interval = long_term_scaling_interval
        self.last_updated_time = -1e9
        self.last_reported_allocation_time = 0
        self.allocation_interval = 600
        self.max_time = max_time

    def scaling_interval_reached(self) -> bool:
        return clock() >= self.last_updated_time + self.long_term_scaling_interval


class MilpGlobalArbiter(GlobalArbiter):
    """MILP based global router"""

    def __init__(self,arima_traces,ilp_output_dir,long_term_scaling_interval,post_processing_strategy,max_time,arima_aware_arbiter):
        super().__init__(long_term_scaling_interval, max_time)
        self.arima_traces = arima_traces
        self.ilp_output_dir = ilp_output_dir
        self.forecast_df = {}
        self.region_clusters = None
        self.region_routers = None
        self.milp_allocator = None
        self.post_processing_strategy = post_processing_strategy
        self.num_instances_log = None
        self.arima_aware_arbiter = arima_aware_arbiter

    def add_region_routers(self, region_routers) -> None:
        self.region_routers = region_routers

    def add_region_clusters(self, region_clusters) -> None:
        self.region_clusters = region_clusters

    def save_results(self):
        utils.save_dict_as_csv(self.num_instances_log, "global_ariber_logs.csv")

    def load_predicted_traces(self) -> None:
        if self.region_clusters is None:
            raise RuntimeError("Region clusters have not been added before loading predicted traces.")

        if self.region_routers is None:
            raise RuntimeError("Region routers have not been added before loading predicted traces.")

        self.regions = len(self.region_clusters)
        self.forecast_df = {}
        unique_models = set()

        for region_id in self.region_clusters.keys():
            self.forecast_df[region_id] = {}

            if region_id not in self.region_routers:
                raise KeyError(f"No region router found for region {region_id}.")

            for model_name in self.region_routers[region_id].model_endpoint_routers.keys():
                unique_models.add(model_name)
                filepath = os.path.join(self.arima_traces, f"forecast_{model_name}_region{region_id}.csv")

                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"ARIMA forecast not found: {filepath}")

                df = pd.read_csv(filepath)

                required_columns = {"Time", "Predicted"}
                missing_columns = required_columns - set(df.columns)

                if missing_columns:
                    raise ValueError(f"Invalid ARIMA forecast file {filepath}. Missing columns: {sorted(missing_columns)}")

                if df.empty:
                    raise ValueError(f"ARIMA forecast file is empty: {filepath}")

                if df["Time"].isna().any() or df["Predicted"].isna().any():
                    raise ValueError(f"ARIMA forecast file contains NaN values: {filepath}")

                if not pd.api.types.is_numeric_dtype(df["Time"]):
                    raise ValueError(f"'Time' must be numeric in {filepath}")

                if not pd.api.types.is_numeric_dtype(df["Predicted"]):
                    raise ValueError(f"'Predicted' must be numeric in {filepath}")

                df["Predicted"] = df["Predicted"].clip(lower=0)
                df = df.sort_values("Time").reset_index(drop=True)

                self.forecast_df[region_id][model_name] = df

                logging.info("Loaded ARIMA forecast: region=%s model=%s rows=%s file=%s", region_id, model_name, len(df), filepath)

        self.model_list = sorted(unique_models)
        self.models = len(self.model_list)
        self.model_to_idx_mapping = {model_name: i for i, model_name in enumerate(self.model_list)}

        logging.info("Models discovered: %s", self.model_list)
        logging.info("Number of regions: %s", self.regions)
        logging.info("Number of models: %s", self.models)

        model_tps = [
            [30 * 7516.67],
            [30 * 5398.07],
            [30 * 1600000],
            [30 * 1600000],
        ]

        if self.models > len(model_tps):
            raise ValueError(f"Found {self.models} models but only {len(model_tps)} model TPS values are configured.")

        self.milp_allocator = MilpLongTermAllocation(
            models=self.models,
            regions=self.regions,
            gpus=1,
            model_interchange_time=[[150] for _ in range(self.models)],
            model_tps=model_tps[:self.models],
            gpu_cost=[10],
        )

        self.num_instances_log = {f"region_{i}_model_{j}": [] for i in range(self.regions) for j in range(self.models)}
        self.num_instances_log["time"] = []

    def current_allocations(self) -> List[List[List[int]]]:
        current_allocations = {}

        for region_id, region_router in self.region_routers.items():
            cur_region = [[0] for _ in range(self.models)]

            for model_name in region_router.model_endpoint_routers.keys():
                model_idx = self.model_to_idx_mapping[model_name]
                val = region_router.model_endpoint_routers[model_name].total_instances
                if val is None:
                    val = 0
                cur_region[model_idx][0] += val

            for model_name in region_router.model_endpoint_routers.keys():
                model_idx = self.model_to_idx_mapping[model_name]
                self.num_instances_log[f"region_{region_id}_model_{model_idx}"].append(cur_region[model_idx][0])

            current_allocations[region_id] = cur_region

        self.num_instances_log["time"].append(clock())
        return [current_allocations[k] for k in sorted(current_allocations.keys())]

    def start_with_8_each(self) -> None:
        ca = self.current_allocations()
        actions = []

        for region in range(self.regions):
            for model in range(self.models):
                while ca[region][model][0] > 8:
                    actions.append((False, 0, region, model, 0))
                    ca[region][model][0] -= 1

        self.schedul_scaling_events(actions)
        logging.info("Start state: %s", self.current_allocations())

    def get_ilp_forecast(self):
        cur_time = clock()

        current_allocation = self.current_allocations()
        current_allocation_transposed = [[[0] for _ in range(self.regions)] for _ in range(self.models)]

        for model in range(self.models):
            for region in range(self.regions):
                current_allocation_transposed[model][region][0] = current_allocation[region][model][0]

        forecast_start = cur_time + 20 * 60
        forecast_end = cur_time + 60 * 60

        logging.info("Building ILP forecast at time %.2f", cur_time)
        logging.info("Forecast window: %.2f -> %.2f", forecast_start, forecast_end)

        tps_forecast = {}

        for region_id, region_router in self.region_routers.items():
            cur_region = [[0] for _ in range(self.models)]

            for model_name in region_router.model_endpoint_routers.keys():
                model_idx = self.model_to_idx_mapping[model_name]
                df = self.forecast_df[region_id][model_name]

                forecast_window = df[(df["Time"] >= forecast_start) & (df["Time"] <= forecast_end)]

                if forecast_window.empty:
                    logging.warning("No ARIMA forecast available for region=%s model=%s at current time=%.2f", region_id, model_name, cur_time)
                    continue

                max_prediction = forecast_window["Predicted"].max()
                cur_region[model_idx][0] = max_prediction

                logging.info("Max predicted TPS: region=%s model=%s prediction=%.2f", region_id, model_name, max_prediction)

            tps_forecast[region_id] = cur_region

        tps_forecast_final = [[[0] for _ in range(self.regions)] for _ in range(self.models)]

        for model in range(self.models):
            for region in range(self.regions):
                tps_forecast_final[model][region][0] = tps_forecast[region][model][0]

        logging.info("Final TPS forecast: %s", tps_forecast_final)

        return self.milp_allocator.get_ilp_allocations(
                    current_allocation_transposed,
                    tps_forecast_final,
                    self.ilp_output_dir
                )

    def post_process_ilp(self, ilp_forecast: List[List[List[int]]]) -> List[Tuple[bool, int, int, int, int]]:
        actions = []
        indices = [(i, j, k) for i in range(self.regions) for j in range(self.models) for k in range(1)]

        if self.post_processing_strategy == "immediate":
            for region, model, gpu in indices:
                change = ilp_forecast[region][model][gpu]

                if change < 0:
                    for _ in range(-change):
                        actions.append((False, 0, region, model, gpu))

            for region, model, gpu in indices:
                change = ilp_forecast[region][model][gpu]

                if change > 0:
                    for _ in range(change):
                        actions.append((True, 0, region, model, gpu))

        elif self.post_processing_strategy == "evenly_distributed":
            raise NotImplementedError(f"ILP post processing strategy {self.post_processing_strategy} not implemented")
        else:
            raise NotImplementedError(f"ILP post processing strategy {self.post_processing_strategy} not implemented")

        return actions

    def schedul_scaling_events(self, changes: List[Tuple[bool, int, int, int, int]]) -> None:
        for action, delay, region, model, gpu in changes:
            if clock() + delay >= self.max_time:
                continue

            if action:
                f = lambda self=self, region=region, model=model: self.region_clusters[region].arbiter.force_scale_up(self.region_routers[region].model_endpoint_routers[self.model_list[model]])
            else:
                f = lambda self=self, region=region, model=model: self.region_clusters[region].arbiter.force_scale_down(self.region_routers[region].model_endpoint_routers[self.model_list[model]])

            schedule_event(delay, f)

    def scale(self) -> None:
        ilp_forecast = self.get_ilp_forecast()
        ilp_changes = self.post_process_ilp(ilp_forecast)

        logging.info("Current allocation at time %s: %s", clock(), self.current_allocations())
        logging.info("ILP forecast at time %s: %s", clock(), ilp_forecast)

        for region_cluster in self.region_clusters.values():
            region_cluster.arbiter.reset_changes()

        self.schedul_scaling_events(ilp_changes)
        self.last_updated_time = clock()