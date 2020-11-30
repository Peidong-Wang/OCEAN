# OCEAN: Operation Counter for Efficient Audio/Speech Processing Networks


## Acknowledgement

This repo is inspired by and largely dependent on THOP: https://github.com/Lyken17/pytorch-OpCounter.git. We focus on the evaluation of audio/speech processing models.

## Slack Group

https://join.slack.com/t/ocean-zdz3431/shared_invite/zt-jns0uib8-7nEYX8HNq0eq3LYq2Lj8rw

## How to install 
    
`python setup.py install`

<!-- `pip install thop` (now continously intergrated on [Github actions](https://github.com/features/actions))

OR

`pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git` -->
    
## How to use 
* Basic usage 
    ```python
    from torchvision.models import resnet50
    from ocean import profile
    model = resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))
    ```    

* Define the rule for 3rd party module.
    ```python
    class YourModule(nn.Module):
        # your definition
    def count_your_model(model, x, y):
        # your rule here
    
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ), 
                            custom_ops={YourModule: count_your_model})
    ```
    
* Improve the output readability

    Call `ocean.clever_format` to give a better format of the output.
    ```python
    from ocean import clever_format
    macs, params = clever_format([macs, params], "%.3f")
    ```    
