import torch
from ocean import profile_macs
from utils.load_trained_model import load_trained_model


# Load model.
model, train_args = load_trained_model("model.acc.best")  # Use a trained model the same as those packed in ESPnet.

# Define sample inputs.
inputs = torch.randn(1, 10, 83)
ilens = torch.tensor([10], dtype=torch.long)
ys_pad = torch.tensor([[0]], dtype=torch.long)

# Profile the model.
model.eval()
macs = profile_macs(model, (inputs, ilens, ys_pad))

# Print MACs.
print("MACs: {}".format(macs))
