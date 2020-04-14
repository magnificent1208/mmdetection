import torch


def weight_quantize_parameter(weight, bits=8):
    max_w = weight.max().item()
    min_w = weight.min().item()
    level = 2 ** bits - 1
    scale = (max_w - min_w) / level
    zero_point = round((0.0 - min_w) / scale)
    return scale, zero_point


def quantize(weight, S, Z, bits=8):
    return torch.clamp((weight / S).round() + Z, 0, 2 ** bits - 1)


def unquantize(weight, S, Z):
    return S * (weight - Z)


def quantize_net(net, bits=6):
    for n, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            s, z = weight_quantize_parameter(module.weight, bits)
            module.weight = torch.nn.Parameter(unquantize(quantize(module.weight, s, z, bits), s, z))
            if module.bias is not None:
                s, z = weight_quantize_parameter(module.bias, bits)
                module.bias = torch.nn.Parameter(unquantize(quantize(module.bias, s, z, bits), s, z))
