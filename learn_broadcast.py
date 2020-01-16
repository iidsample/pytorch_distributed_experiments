import torch
import torch.distributed as dist
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument("--master-ip", type=str)
parser.add_argument("--world-size", type=int)
args = parser.parse_args()
dist.init_process_group(backend="nccl", init_method=args.master_ip,
                        world_size=args.world_size, rank=args.local_rank)

test_tensor = torch.zeros((4,4), device='cuda:0')

print (test_tensor)
if args.local_rank == 0:
    test_tensor[0,0] = 4
    test_tensor[1,1] = 2

torch.distributed.broadcast(test_tensor, 0)
print (test_tensor)
