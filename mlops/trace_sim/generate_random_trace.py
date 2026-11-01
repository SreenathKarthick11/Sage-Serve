"""Generate a random (static, non-streaming) trace and store it as a new immutable version
under  data/static/raw/<version>/.

    python -m trace_sim.generate_random_trace --num-requests 5000 --seed 1

The random stream is seeded, so (num_requests, seed) fully determines the file: running it
twice gives byte-identical output and therefore the same version id.

Header: request_id,batch_id,client_tenant,request_type,scenario,sla,utility,regions,model_type,
        workload_type,application_id,arrival_timestamp,batch_size,prompt_size,token_size
"""
import argparse
import random
import shutil

from pipeline.versioning import finalize, new_tmp_dir, sha256_file

COLUMNS = ("request_id,batch_id,client_tenant,request_type,scenario,sla,utility,regions,model_type,"
           "workload_type,application_id,arrival_timestamp,batch_size,prompt_size,token_size")


def generate(num_requests, seed, out_path):
    """Write the trace to `out_path`; returns the number of data rows written."""
    rng = random.Random(seed)
    batch_idx = 0
    num_regions = 3

    with open(out_path, 'w') as f:
        f.write(COLUMNS + "\n")
        for request_id in range(num_requests):
            client_tenant = rng.randint(0, num_regions)
            request_type = 2
            scenario = "EnterpriseSydney"
            random_num = rng.random()
            model_type = rng.choice(["A", "B", "C", "D"])
            application_id = 0
            regions = ''.join(rng.sample("012", 3))
            if random_num < 0.7:
                workload_type = "prod"
                batch_id  = -1
                sla = 10
                utility = 3
                batch_size = 1
                arrival_timestamp = request_id
                prompt_size = rng.randint(1, 1024)
                token_size = rng.randint(1, 1024)
                f.write(f"{request_id},{batch_id},{client_tenant},{request_type},{scenario},"\
                        f"{sla},{utility},{regions},{model_type},{workload_type},"\
                        f"{application_id},{arrival_timestamp},{batch_size},{prompt_size},{token_size}\n")
            elif random_num < 0.8:
                workload_type = "prod"
                batch_id  = -1
                sla = 60 * 60
                utility = 2
                batch_size = 1
                arrival_timestamp = request_id
                prompt_size = rng.randint(1, 1024)
                token_size = rng.randint(1, 1024)
                f.write(f"{request_id},{batch_id},{client_tenant},{request_type},{scenario},"\
                        f"{sla},{utility},{regions},{model_type},{workload_type},"\
                        f"{application_id},{arrival_timestamp},{batch_size},{prompt_size},{token_size}\n")
            elif random_num < 0.9:
                workload_type = "prod" if rng.random() < 0.5 else "dev"
                batch_id = batch_idx
                batch_idx += 1
                sla = 24 * 60 * 60
                utility = 1
                batch_size = 128
                arrival_timestamp = request_id
                for _ in range(batch_size):
                    prompt_size = rng.randint(1, 1024)
                    token_size = rng.randint(1, 1024)
                    f.write(f"{request_id},{batch_id},{client_tenant},{request_type},{scenario},"\
                            f"{sla},{utility},{regions},{model_type},{workload_type},"\
                            f"{application_id},{arrival_timestamp},{batch_size},{prompt_size},{token_size}\n")
            else:
                workload_type = "dev"
                batch_id  = -1
                sla = -1
                utility = 0
                batch_size = 1
                arrival_timestamp = request_id
                prompt_size = rng.randint(1, 1024)
                token_size = rng.randint(1, 1024)
                f.write(f"{request_id},{batch_id},{client_tenant},{request_type},{scenario},"\
                        f"{sla},{utility},{regions},{model_type},{workload_type},"\
                        f"{application_id},{arrival_timestamp},{batch_size},{prompt_size},{token_size}\n")

    with open(out_path) as f:
        return sum(1 for _ in f) - 1


def ingest(num_requests=5000, seed=42):
    """Create one new version of the static raw trace. Returns its directory."""
    tmp_dir = new_tmp_dir("static", "raw")
    try:
        trace = tmp_dir / "trace.csv"
        rows = generate(num_requests, seed, trace)
        params = {"num_requests": num_requests, "seed": seed, "generator": "generate_random_trace"}
        details = {"ingest": "python-random", "status": "complete", "rows": rows}
        return finalize(tmp_dir, "static", "raw", sha256_file(trace)[:12], parents=[], params=params, details=details)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-requests", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    final = ingest(a.num_requests, a.seed)
    print(f"wrote {final}")


if __name__ == "__main__":
    main()
