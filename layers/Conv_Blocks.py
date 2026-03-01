import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    
class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        
        if isinstance(kernel_size, int):
            kernel_size = (1, kernel_size)   
        if isinstance(dilation, int):
            dilation = (1, dilation)
        else:       
            dilation = (1, dilation[0]) if len(dilation) == 1 else (dilation[0], dilation[1])
        
        
        padding = (0, (kernel_size[1] - 1) * dilation[1]) 
        
        super(CausalConv2d, self).__init__(
            in_channels, out_channels, kernel_size, 
            padding=padding, 
            dilation=dilation, 
            **kwargs
        )
    
    def forward(self, x):
        result = super().forward(x)  
        if result.size(3) > x.size(3):  
            result = result[:, :, :, -x.size(3):] 
        return result

class TCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rates=[1, 2, 4, 8], dropout=0.1):
        super(TCN_Module, self).__init__()
        self.layers = nn.ModuleList()
        
        for i, dilation in enumerate(dilation_rates):
            self.layers.append(nn.Sequential(
                CausalConv2d(in_channels, out_channels, kernel_size=(1, kernel_size), 
                            dilation=(1, dilation)),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
