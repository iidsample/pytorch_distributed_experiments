import torch
import torch.distributed as dist

os.environ["MASTER_ADDR"] = "192.168.0.5"
os.environ["MASTER_PORT"] = "6066"
dist.init_process_group(backend="nccl", init_method="file:///home/saurabh/pytorch_distributed_experiments/distributed_test",
                        world_size = 2, rank=0)
out_tensor_list = [[torch.FloatTensor([0]).cuda(0),
                    torch.FloatTensor([0]).cuda(0)]]
temp_tensor = torch.FloatTensor([0.5]).cuda(0)
tensor_list = list()
tensor_list.append(temp_tensor)
dist.all_gather_multigpu(out_tensor_list, tensor_list)
print(tensor_list)
print(out_tensor_list)
