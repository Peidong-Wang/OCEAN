# OCEAN: Operation Counter for Efficient Audio/Speech Processing Networks


## Updates and Acknowledgement

The default version of OCEAN now uses torch.jit.trace as in torchprofile: https://github.com/zhijian-liu/torchprofile. The THOP based version is saved to the *thop* branch.

## How to install 

`python setup.py install`

## How to use 
### Installation verification:

Define PyTorch model and its (dummy) input:

```python
import torch
from torchvision.models import resnet18

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

Measure the number of MACs using `profile_macs`:

```python
from ocean import profile_macs

macs = profile_macs(model, inputs)
```

### Examples:

Check out the `egs/` folder.

### Register handlers:

For warnings like `UserWarning: No handlers found: "aten::<a specific operation>". Skipped.`, please register the missing handler to `ocean/handlers.py`.

## Slack Group

https://join.slack.com/t/ocean-zdz3431/shared_invite/zt-jns0uib8-7nEYX8HNq0eq3LYq2Lj8rw
