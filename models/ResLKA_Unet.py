import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

def replace_bn_with_gn(module, num_groups=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups_adj = min(num_groups, num_channels)
            while num_channels % num_groups_adj != 0 and num_groups_adj > 1:
                num_groups_adj -= 1
            gn = nn.GroupNorm(num_groups=num_groups_adj, num_channels=num_channels)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)
    return module

class conv_block(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels=num_filters, 
                kernel_size=3, 
                stride=1, 
                padding='same',
                bias=False
            ), 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters, 
                out_channels=num_filters, 
                kernel_size=3, 
                stride=1, 
                padding='same',
                bias=False
            ),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_block(x)

class decoder_block(nn.Module):
    def __init__(self, block, in_channels, out_channels, skip_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=2, 
            stride=2,
            bias=False
        )
        self.conv_block = block(in_channels=out_channels+skip_channels,num_filters=out_channels)

    def forward(self, x, skip_features):
        x = self.deconv(x)
        x = torch.cat((x, skip_features), 1)
        x = self.conv_block(x)
        return x
    
class LargeKernelAttn(nn.Module):
    def __init__(self,
                 channels):
        super(LargeKernelAttn, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False
        )
        self.dwdconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            padding=9,
            groups=channels,
            dilation=3,
            bias=False
        )
        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        weight = self.pwconv(self.dwdconv(self.dwconv(x)))

        return x * weight

class ResLKA_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )

        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.LKA = LargeKernelAttn(channels=2048)
        

        self.decoder1 = decoder_block(conv_block, in_channels=2048, out_channels=1024, skip_channels=1024)
        self.decoder2 = decoder_block(conv_block, in_channels=1024, out_channels=512, skip_channels=512)
        self.decoder3 = decoder_block(conv_block, in_channels=512, out_channels=256, skip_channels=256)
        self.decoder4 = decoder_block(conv_block, in_channels=256, out_channels=64, skip_channels=64)        

        self.up_final = nn.ConvTranspose2d(
            64, 
            64, 
            kernel_size=2, 
            stride=2,
            bias=False
        ) 
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        replace_bn_with_gn(self, num_groups=16)
        self._init_weights()

    def _init_weights(self):
        backbone_modules = {
            id(self.conv1),
            id(self.maxpool),
            id(self.encoder1),
            id(self.encoder2),
            id(self.encoder3),
            id(self.encoder4),
        }

        for m in self.modules():
            if id(m) in backbone_modules:
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        x1 = self.conv1(x)      
        x2 = self.maxpool(x1)   
        x3 = self.encoder1(x2)  
        x4 = self.encoder2(x3)  
        x5 = self.encoder3(x4)  
        x6 = self.encoder4(x5)  
        x6 = self.LKA(x6)

        d1 = self.decoder1(x6, x5)  
        d2 = self.decoder2(d1, x4)  
        d3 = self.decoder3(d2, x3)  
        d4 = self.decoder4(d3, x1)  

        up_final = self.up_final(d4)  
        output = self.final_conv(up_final)  
        return self.sigmoid(output)


