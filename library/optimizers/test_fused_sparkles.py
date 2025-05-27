import torch
from sparkles import SPARKLES

# Create a simple linear model in bfloat16 on CUDA
d_in, d_out = 1000, 10
model = torch.nn.Linear(d_in, d_out, device='cuda', dtype=torch.bfloat16)
optimizer = SPARKLES(model.parameters(), lr=1e-2)

# Dummy input and target
data = torch.randn(16, d_in, device='cuda', dtype=torch.bfloat16)
target = torch.randn(16, d_out, device='cuda', dtype=torch.bfloat16)

# Forward pass
output = model(data)
loss = torch.nn.functional.mse_loss(output, target)

# Backward pass
torch.cuda.synchronize()
loss.backward()

# Optimizer step
torch.cuda.synchronize()
optimizer.step()
torch.cuda.synchronize()

print("SPARKLES step with fused CUDA kernel completed successfully.") 