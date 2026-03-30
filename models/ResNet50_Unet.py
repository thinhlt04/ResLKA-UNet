import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

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
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters, 
                out_channels=num_filters, 
                kernel_size=3, 
                stride=1, 
                padding='same',
                bias=True
            ),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_block(x)

class encoder_block(nn.Module):
    def __init__(self, block, in_channels, num_filters):
        super().__init__()
        self.conv_block = block(in_channels = in_channels, num_filters=num_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        s = self.conv_block(X)
        p = self.pool(s)
        return s, p 

class decoder_block(nn.Module):
    def __init__(self, block, in_channels, out_channels, skip_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=2, 
            stride=2,
            bias=True
        )
        self.conv_block = block(in_channels=out_channels+skip_channels,num_filters=out_channels)

    def forward(self, x, skip_features):
        x = self.deconv(x)
        x = torch.cat((x, skip_features), 1)
        x = self.conv_block(x)
        return x

class Resnet50_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.input_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=3, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=True
        )

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

        self.decoder1 = decoder_block(conv_block, in_channels=2048, out_channels=1024, skip_channels=1024)
        self.decoder2 = decoder_block(conv_block, in_channels=1024, out_channels=512, skip_channels=512)
        self.decoder3 = decoder_block(conv_block, in_channels=512, out_channels=256, skip_channels=256)
        self.decoder4 = decoder_block(conv_block, in_channels=256, out_channels=64, skip_channels=64)        

        self.up_final = nn.ConvTranspose2d(
            64, 
            64, 
            kernel_size=2, 
            stride=2,
            bias=True
        ) 
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()


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
        x = self.input_conv(x)  
        x1 = self.conv1(x)      
        x2 = self.maxpool(x1)   
        x3 = self.encoder1(x2)  
        x4 = self.encoder2(x3)  
        x5 = self.encoder3(x4)  
        x6 = self.encoder4(x5)  

        d1 = self.decoder1(x6, x5)  
        d2 = self.decoder2(d1, x4)  
        d3 = self.decoder3(d2, x3)  
        d4 = self.decoder4(d3, x1)  

        up_final = self.up_final(d4)  
        output = self.final_conv(up_final)  
        return self.sigmoid(output)