import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f"ori_x: {x.dtype}")
        x = self.fc1(x)
        print(f"after_fc1: {x.dtype}")
        x = self.relu(x)
        print(f"after_relu: {x.dtype}")
        x = self.ln(x)
        print(f"after_ln: {x.dtype}")
        x = self.fc2(x)
        print(f"after_fc2: {x.dtype}")
        return x

model : torch.nn.Module = ToyModel(in_features=32,out_features=10).to("cuda")
dtype : torch.dtype = torch.bfloat16
x : torch.Tensor = torch.randn(128, 32, dtype=torch.float32).to("cuda")

with torch.autocast(device_type="cuda",dtype = dtype):
    y = model(x)
    