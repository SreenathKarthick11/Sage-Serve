import logging

from abc import ABC, abstractmethod
from typing import List, Tuple
import datetime

from simulator import clock, schedule_event, cancel_event, reschedule_event
import start_state_repo
import utils

from pulp import *

class LongTermAllocation(ABC):
    def __init__(self, models: int, regions: int, gpus: int) -> None:
        self.models = models
        self.regions = regions
        self.gpus = gpus

    @abstractmethod
    def get_allocation(self,
                       timeframe: int, 
                       interactive_forecast: List[List[List[int]]],
                       non_interactive_forecast: List[List[List[int]]],
                       opportunistic_forecast: List[List[List[int]]],
                       current_allocation: List[List[List[int]]]) -> List[List[List[int]]]:
        pass

class MilpLongTermAllocation(LongTermAllocation):
    def __init__(self, 
                 models: int, 
                 regions: int, 
                 gpus: int, 
                 model_interchange_time: List[List[float]], 
                 model_tps: List[List[float]],
                 gpu_cost: List[float],
                 BIG_M: float = 1e6) -> None:
        super().__init__(models, regions, gpus)
        self.model_interchange_time = model_interchange_time
        self.model_tps = model_tps
        self.gpu_cost = gpu_cost
        self.BIG_M = BIG_M
        assert len(self.model_interchange_time) == self.models, f"Model interchange time should have dimensions {models}x{gpus}, got {len(self.model_interchange_time)} as first dimension"
        for i in range(self.models):
            assert len(self.model_interchange_time[i]) == self.gpus, f"Model interchange time should have dimensions {models}x{gpus}, got {len(self.model_interchange_time[i])} as second dimension at index {i}"
        assert len(self.model_tps) == self.models, f"Model TPS should have dimensions {models}x{gpus}, got {len(self.model_tps)} as first dimension"
        for i in range(self.models):
            assert len(self.model_tps[i]) == self.gpus, f"Model TPS should have dimensions {models}x{gpus}, got {len(self.model_tps[i])} as second dimension at index {i}" 
        assert len(self.gpu_cost) == self.gpus, f"GPU cost should have dimensions {gpus}, got {len(self.gpu_cost)} as first dimension"
    
    def get_ilp_allocations(self, 
                            current_allocation: List[List[List[int]]], 
                            forecast_demand: List[List[int]],
                            path: str,
                            model_tps: List[List[float]]=None,
                            model_startup_cost: List[List[float]]=None,
                            gpu_cost: List[float]=None
                            ) -> List[List[List[int]]]:
        if model_tps == None:
            model_tps = self.model_tps
        if model_startup_cost == None:
            model_startup_cost = self.model_interchange_time
        if gpu_cost == None:
            gpu_cost = self.gpu_cost
        prob = LpProblem(f"solver_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", LpMinimize)
        indices = [(i, j, k) for i in range(self.models) for j in range(self.regions) for k in range(self.gpus)]
        allocation_var = LpVariable.dicts("allocations", indices, cat="Integer")
        increase_signal_r = LpVariable.dicts("IS_r", indices, cat="Integer")
        increase_signal_s = LpVariable.dicts("IS_s", indices, cat="Integer")
        new_allocation_cost = LpVariable.dicts("new_allocation_cost", indices, cat="Integer")

        # Constraints
        # dont deallocate more than you have
        for idx in indices:
            allocation_var[idx].setInitialValue(-current_allocation[idx[0]][idx[1]][idx[2]])
            prob += allocation_var[idx] >= -current_allocation[idx[0]][idx[1]][idx[2]]
         # always satisfy low latency TPS
        for i in range(self.models):
            for j in range(self.regions):
                prob += lpSum(
                            [(allocation_var[(i, j, k)] + current_allocation[i][j][k]) * model_tps[i][k] for k in range(self.gpus)]) >= forecast_demand[i][j] 
        # check increase signal variable
        for idx in indices:
            prob += 1 >= increase_signal_r[idx] >= 0
            prob += 1 >= increase_signal_s[idx] >= 0
            prob += increase_signal_s[idx] + increase_signal_r[idx] == 1
            prob += -self.BIG_M * increase_signal_r[idx] + increase_signal_s[idx] <= allocation_var[idx] 
            prob += allocation_var[idx] <= self.BIG_M * increase_signal_s[idx]
            prob += new_allocation_cost[idx] <= increase_signal_s[idx] * self.BIG_M
            prob += new_allocation_cost[idx] >= -increase_signal_s[idx] * self.BIG_M
            prob += new_allocation_cost[idx] <= allocation_var[idx] + (1 - increase_signal_s[idx]) * self.BIG_M
            prob += new_allocation_cost[idx] >= allocation_var[idx] - (1 - increase_signal_s[idx]) * self.BIG_M

        # objective function 
        prob += (
            lpSum([allocation_var[idx] * gpu_cost[idx[2]] + new_allocation_cost[idx] * model_startup_cost[idx[0]][idx[2]]] for idx in indices),
            "Cost of changing GPUs"
        )
        if path:
            prob.writeLP(f"{path}/../../ilp_outputs/my_output_at_{clock()}.lp")
        prob.solve(PULP_CBC_CMD(msg=0))
        prob.roundSolution()

        ret_val = [[[0 for ___ in range(self.gpus)] for _ in range(self.models)] for __ in range(self.regions)]
        for i in range(self.regions):
            for j in range(self.models):
                for k in range(self.gpus):
                    ret_val[i][j][k] = allocation_var[(j, i, k)].value()
        return ret_val

    def get_allocation(self,
                       timeframe: int, 
                       interactive_forecast: List[List[List[int]]],
                       non_interactive_forecast: List[List[List[int]]],
                       opportunistic_forecast: List[List[List[int]]],
                       current_allocation: List[List[List[int]]]) -> List[List[List[int]]]:
        forecast = []
        for i in range(self.models):
            model_wise_forecast = []
            for j in range(self.regions):
                model_wise_forecast.append(interactive_forecast[i][j][-1] + interactive_forecast[i][j][-2])
            forecast.append(model_wise_forecast)
        allocation = self.get_ilp_allocations(current_allocation, forecast, None, self.model_tps, self.model_interchange_time, self.gpu_cost)
        return allocation

if __name__ == "__main__":
    models = 2 # n
    regions = 2 # m
    gpus = 1 # g
    mit = [[1], [1]] # n x g
    tps = [[100], [100]] # n x g
    gpu_cost = [10] # g
    allocator = MilpLongTermAllocation(models, regions, gpus, mit, tps, gpu_cost)
    t = 2
    forecast = [
        [[0, 101], [0, 10]],
        [[0, 10], [0, 201]]
    ] # n x m x t
    ca = [
        [[1], [1]],
        [[1], [1]]
    ] # n x m x g
    print(allocator.get_allocation(t, forecast, forecast, forecast, ca))