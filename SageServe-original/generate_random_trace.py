import os
import random
# request_id,batch,id,client_tenant,request_type,scenario,sla,utility,regions,model_type,workload_type,application_id,arrival_timestamp,batch_size,prompt_size,token_size

def main():
    num_requests = 1000
    batch_idx = 0
    num_regions = 3
    with open('traces/random_trace.csv', 'w') as f:
        f.write(f"request_id,batch_id,client_tenant,request_type,scenario,sla,utility,regions,model_type,workload_type,application_id,arrival_timestamp,batch_size,prompt_size,token_size\n")
        for request_id in range(num_requests):
            client_tenant = random.randint(0, num_regions)
            request_type = 2
            scenario = "EnterpriseSydney"
            random_num = random.random()
            model_type = random.choice(["llama2-70b", "bloom-176b"])
            application_id = 0
            regions = ''.join(random.sample("012", 3))
            if random_num < 0.7:
                workload_type = "prod"
                batch_id  = -1
                sla = 10
                utility = 3
                batch_size = 1
                arrival_timestamp = request_id
                prompt_size = random.randint(1, 1024)
                token_size = random.randint(1, 1024)
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
                prompt_size = random.randint(1, 1024)
                token_size = random.randint(1, 1024)
                f.write(f"{request_id},{batch_id},{client_tenant},{request_type},{scenario},"\
                        f"{sla},{utility},{regions},{model_type},{workload_type},"\
                        f"{application_id},{arrival_timestamp},{batch_size},{prompt_size},{token_size}\n")
            elif random_num < 0.9:
                workload_type = "prod" if random.random() < 0.5 else "dev"
                batch_id = batch_idx
                batch_idx += 1
                sla = 24 * 60 * 60
                utility = 1
                batch_size = 128
                arrival_timestamp = request_id
                for _ in range(batch_size):
                    prompt_size = random.randint(1, 1024)
                    token_size = random.randint(1, 1024)
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
                prompt_size = random.randint(1, 1024)
                token_size = random.randint(1, 1024)
                f.write(f"{request_id},{batch_id},{client_tenant},{request_type},{scenario},"\
                        f"{sla},{utility},{regions},{model_type},{workload_type},"\
                        f"{application_id},{arrival_timestamp},{batch_size},{prompt_size},{token_size}\n")

if __name__ == "__main__":
    main()