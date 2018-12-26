import torch
import torch.distributed as dist
import os

os.environ["MASTER_ADDR"] = "192.168.0.5"
os.environ["MASTER_PORT"] = "6066"
dist.init_process_group(backend="nccl", init_method="file:///home/saurabh/pytorch_distributed_experiments/distributed_test",
                        world_size = 2, rank=1)
out_tensor_list = [[torch.FloatTensor([0]).cuda(1),
                    torch.FloatTensor([0]).cuda(1)]]

temp_tensor = torch.FloatTensor([1]).cuda(1)
tensor_list = list()
tensor_list.append(temp_tensor)
dist.all_gather_multigpu(out_tensor_list, tensor_list)
print(tensor_list)
print(out_tensor_list)
