import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
        
def convert_bn_to_ln(model):
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            ln = LayerNorm(num_features, 1e-6)
            if module.affine:
                ln.weight.data.copy_(module.weight.data / torch.sqrt(module.running_var + module.eps))
                ln.bias.data.copy_(module.bias.data - module.running_mean * module.weight.data / torch.sqrt(module.running_var + module.eps))
            setattr(model.model, name, ln)
            breakpoint()
            # model._modules[name] = ln

            # setattr(module, name, ln)
    return model
    # if isinstance(module, nn.BatchNorm2d):
    #     print(module)
    #     num_features = module.num_features
    #     ln = LayerNorm(num_features, module.eps)
    #     if module.affine:
    #         ln.weight.data.copy_(module.weight.data / torch.sqrt(module.running_var + module.eps))
    #         ln.bias.data.copy_(module.bias.data - module.running_mean * module.weight.data / torch.sqrt(module.running_var + module.eps))
    #     setattr(module, name, ln)
    #     return ln
    # else:
    #     return module

def replace_bn_with_custom(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            eps = module.eps
            ln = LayerNorm(num_features, module.eps, data_format="channels_first")
            parent_module = get_parent_module(model, name)
            setattr(parent_module, name.split(".")[-1], ln)

def get_parent_module(model, name):
    parent_module = model
    name_split = name.split(".")
    for sub_name in name_split[:-1]:
        parent_module = getattr(parent_module, sub_name)
    return parent_module