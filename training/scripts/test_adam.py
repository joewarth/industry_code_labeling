import torch

torch.set_num_threads(1)

print("torch version:", torch.__version__, flush=True)

w = torch.nn.Parameter(torch.randn(3, 3))

print("testing tiny Adam", flush=True)
opt = torch.optim.Adam([w], lr=1e-3)
print("tiny Adam built", flush=True)