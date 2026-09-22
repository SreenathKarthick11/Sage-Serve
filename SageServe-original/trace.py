import pandas as pd
import time
import numpy as np

from request import Request
from simulator import clock

import logging


class Trace():
    def __init__(self, csv_path, siloed):
        self.csv_path = csv_path
        self.siloed = siloed
        print(siloed)
        df = pd.read_csv(csv_path)

        self.num_requests = len(df)
        self.batch_size = 1<<8

        # batching
        min_value = df['arrival_timestamp'].min()
        max_value = df['arrival_timestamp'].max()
        bins = np.arange(min_value, max_value + self.batch_size, self.batch_size)
        df['batch'] = pd.cut(df['arrival_timestamp'], bins=bins, right=False)
        self.batch_sizes = [len(data) for group, data in df.groupby('batch', observed=False)]
        self.skip = 0
        del df

    def populate_requests(self):
        start_time = time.time()
        if len(self.batch_sizes) == 0:
            return []
        nrows = self.batch_sizes.pop(0)
        df = pd.read_csv(self.csv_path, skiprows=range(1, self.skip+1), nrows=nrows, dtype={'regions': str})
        df["regions"] = df["regions"].apply(lambda x: x if len(x) == 3 else f"0{x}")
        requests = []
        for _, request_dict in df.iterrows():
            request_dict["token_size"] = max(request_dict["token_size"], 1)
            request = Request.from_dict(request_dict)
            if self.siloed:
                request.model_type += "-" + request.workload_type[0]
            requests.append(request)
        logging.info(f"Read {len(requests)} requests in {round(time.time() - start_time, 2)}s at {clock()} with range({df['arrival_timestamp'].min()}, {df['arrival_timestamp'].max()})")
        self.skip += nrows
        del df
        return requests

    @classmethod
    def from_csv(cls, path, siloed):
        return Trace(path, siloed)
 # type: ignore
