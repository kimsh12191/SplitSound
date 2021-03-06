# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py <- 여기서 들고옴
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py <- 이건 윗사람이 들고온곳인듯

import torch.nn as nn
import torch
class CAM8s(nn.Module):
    def __init__(self, n_class=2):
        super(CAM8s, self).__init__()
        # conv1
                
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=32),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/2
        )
        # conv2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/4
        )
        # conv3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/8
        )
        # conv4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/16
        )
        # conv5
        self.conv_block5 = nn.Sequential(
             nn.Conv2d(256, 512, 3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, 3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.Conv2d(512, 512, 3, padding=1),
             nn.BatchNorm2d(512),
             nn.ReLU(inplace=True),
             nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/32
        )
        # fc6
        self.fc_block6 = nn.Sequential(
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # fc7
        self.fc_block7 = nn.Sequential(
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        # cam filter 
        self.cam_feature = nn.Sequential(
            nn.Conv2d(1024, 128, 1),
#             nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, n_class),
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
                    
    def forward(self, x):
        h = x
        h = self.conv_block1(h)

        h = self.conv_block2(h)

        h = self.conv_block3(h)
        pool3 = h  # 1/8

        h = self.conv_block4(h)
        pool4 = h  # 1/16

        h = self.conv_block5(h)

        h = self.fc_block6(h)
        
        h = self.fc_block7(h)
        
        feature = self.cam_feature(h)
#         print (feature.shape)
        n_batch, _, width, height = feature.shape
        agv_feature = nn.AvgPool2d(width, height)(feature)
        
        output = agv_feature.view(n_batch, -1)
        #print(flatten.size())
        output = self.classifier(output)

        
        return output, feature, nn.Sigmoid()(output)