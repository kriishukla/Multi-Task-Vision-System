import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegNet_Encoder(nn.Module):
    def __init__(self, in_chn=3, BN_momentum=0.5):
        super(SegNet_Encoder, self).__init__()
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        # Stage 1
        self.ConvEn11 = nn.Conv2d(in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        # Stage 2
        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        # Stage 3
        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        # Stage 4
        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        # Stage 5
        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        
    def forward(self, x):
        # Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        # Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        # Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        # Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        # Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        
        return x, [ind1, ind2, ind3, ind4, ind5], [size1, size2, size3, size4, size5]

class SegNet_Decoder(nn.Module):
    def __init__(self, out_chn, BN_momentum=0.5):
        super(SegNet_Decoder, self).__init__()
        # Stage 5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, momentum=BN_momentum)

        # Stage 4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(256, momentum=BN_momentum)

        # Stage 3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.conv3_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(128, momentum=BN_momentum)

        # Stage 2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.conv2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64, momentum=BN_momentum)

        # Stage 1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.conv1_2 = nn.Conv2d(64, out_chn, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(out_chn, momentum=BN_momentum)
    
    def forward(self, x, indices, sizes):
        ind1, ind2, ind3, ind4, ind5 = indices
        size1, size2, size3, size4, size5 = sizes

        # Stage 5
        x = self.unpool5(x, ind5, output_size=size4)
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))

        # Stage 4
        x = self.unpool4(x, ind4, output_size=size3)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))

        # Stage 3
        x = self.unpool3(x, ind3, output_size=size2)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))

        # Stage 2
        x = self.unpool2(x, ind2, output_size=size1)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))

        # Stage 1
        x = self.unpool1(x, ind1)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.bn1_2(self.conv1_2(x))
        return x

class SegNet_Pretrained(nn.Module):
    def __init__(self, encoder_weight_pth, out_chn=32, in_chn=3):
        super(SegNet_Pretrained, self).__init__()
        self.encoder = SegNet_Encoder(in_chn=in_chn)
        self.decoder = SegNet_Decoder(out_chn=out_chn)
        print(2)
        # Load pretrained encoder weights.
        state = torch.load(encoder_weight_pth, map_location=device)
        # If the saved state has a key 'weights_only', extract it.
        if 'weights_only' in state:
            print(2)
            encoder_state_dict = state['weights_only']
        else:
            encoder_state_dict = state
            print(2)
        self.encoder.load_state_dict(encoder_state_dict)
        # Freeze encoder parameters.
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x, indices, sizes = self.encoder(x)
        x = self.decoder(x, indices, sizes)
        return x    

class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # Replace the classifier to match the number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        return self.model(x)['out']