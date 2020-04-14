from mmdet.models import ResNet
import torch

net = ResNet(depth=18)
net.eval()

data = torch.rand(1, 3, 32, 32)
net_out = net.forward(data)

for level in net_out:
    print(level.shape)

