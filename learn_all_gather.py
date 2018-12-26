import torch
import torch.distributed as dist
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()

# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "6066"
print ("World size  {}".format(os.environ["WORLD_SIZE"]))
print ("Hello from local rank {}".format(args.local_rank))
dist.init_process_group(backend="nccl", init_method="env://",
                        world_size = int(os.environ["WORLD_SIZE"]), rank=args.local_rank)
# out_tensor_list = [[torch.FloatTensor([0]).cuda(args.local_rank),
                    # torch.FloatTensor([0]).cuda(args.local_rank)]]
out_tensor_list = [[
    torch.FloatTensor([0]).cuda(args.local_rank) for x in
    range(int(os.environ["WORLD_SIZE"]))]]
temp_tensor = torch.rand(1).cuda(args.local_rank)
print("Tensor{} in device {}".format(temp_tensor, args.local_rank))
tensor_list = list()
tensor_list.append(temp_tensor)
dist.all_gather_multigpu(out_tensor_list, tensor_list)
# print(tensor_list)
print("Out tensor {} on {}".format(out_tensor_list, args.local_rank))
