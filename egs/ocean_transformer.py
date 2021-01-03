import torch
from torch.nn.modules.transformer import Transformer
from ocean import profile_macs

if __name__ == '__main__':
    embed_size = 512
    num_tokens = 30

    model = Transformer(embed_size)
    inputs = (
        torch.randn(num_tokens, 1, embed_size),
        torch.randn(num_tokens, 1, embed_size),
    )

    macs = profile_macs(model, inputs)
    print('MACs: {:.4g} G'.format(macs / 1e9))
