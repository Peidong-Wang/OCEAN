import torch
from ocean import profile
from utils.load_trained_model import load_trained_model


model, train_args = load_trained_model("model.acc.best")  # Use a trained model the same as those packed in ESPnet.

input = torch.randn(1, 10, 83)
ilens = torch.tensor([10], dtype=torch.long)
ys_pad = torch.tensor([[0]], dtype=torch.long)

macs, params = profile(model, inputs=(input, ilens, ys_pad, ))

print("macs: " + str(macs))  # TODO: Change to .format string.
