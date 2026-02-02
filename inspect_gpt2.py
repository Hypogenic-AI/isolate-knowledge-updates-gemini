import torch
from transformers import GPT2Model, GPT2Config

config = GPT2Config()
model = GPT2Model(config)
mlp = model.h[0].mlp
print("MLP type:", type(mlp))

x = torch.randn(1, 10, config.n_embd)
out = mlp(x)
print("Output type:", type(out))
print("Output:", out)
