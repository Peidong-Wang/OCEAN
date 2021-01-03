import torch
import torch.nn as nn
from ocean import profile_macs

if __name__ == '__main__':
    in_features = 16
    out_features = 32

    model = nn.Linear(in_features, out_features)
    inputs = torch.randn(1, in_features)

    macs = profile_macs(model, inputs)
    print("MACs: {}".format(macs))
