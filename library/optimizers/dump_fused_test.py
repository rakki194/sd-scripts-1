import torch
from sparkles import SPARKLES

# Dummy model
model = torch.nn.Linear(1000, 1, device='cuda', dtype=torch.bfloat16)
optimizer = SPARKLES(model.parameters(), lr=1e-2)

# Dummy input and loss
inp = torch.randn(10, 1000, device='cuda', dtype=torch.bfloat16)
out = model(inp)
loss = out.sum()
loss.backward()  # This will create .grad in float32 automatically

optimizer.step()
print("SPARKLES step with fused CUDA kernel completed successfully.")