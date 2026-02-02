# Tutorial: Building Your First Model

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This tutorial walks you through creating a simple custom model using the Machine Learning Zoo architecture.

## 1. Setup

Ensure you have installed the library:

```bash
uv pip install -e .
```

## 2. Define a Configuration

Create a file `configs/tutorial_model.yaml`:

```yaml
model:
  name: "tutorial_net"
  backbone:
    type: "mlp"
    input_dim: 10
    hidden_dim: 64
  head:
    type: "classification"
    num_classes: 2
```

## 3. Create the Backbone

If "mlp" doesn't exist, let's create it in `models/backbones/simple_mlp.py`:

```python
import torch.nn as nn
from models.registry import register_backbone

@register_backbone("mlp")
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
```

## 4. Run Inference

Now you can load and use your composed model:

```python
from models.composed import build_model
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load("configs/tutorial_model.yaml")

# Build
model = build_model(cfg.model)

# Run
import torch
input_data = torch.randn(1, 10)
output = model(input_data)
print(f"Propabilities: {torch.softmax(output, dim=1)}")
```

## 5. Next Steps

- Try adding a `training` loop using our `Trainer` class.
- Explore the `presets/` directory for complex examples.
