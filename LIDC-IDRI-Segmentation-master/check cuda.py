import torch, gc
gc.collect()
torch.cuda.empty_cache()

print(torch.cuda.memory_summary(device=None, abbreviated=False))