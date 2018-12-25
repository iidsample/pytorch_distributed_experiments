import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="file:///distributed_test",
                        world_size = 2, rank=0)

temp_tensor = torch.FloatTensor[0.5].cuda(0)
tensor_list = list()
tensor_list.append(temp_tensor)
dist.all_reduce_multigpu(tensor_list)
print(tensor_list)
print(temp_tensor)
