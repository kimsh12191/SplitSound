# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py <- 여기서 들고옴
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py <- 이건 윗사람이 들고온곳인듯

import torch.nn as nn
import torch
class FCN8swtClassify(nn.Module):
    def __init__(self, n_class=2):
        super(FCN8swtClassify, self).__init__()
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
        self.relu    = nn.ReLU(inplace=True)
        self.score_fr = nn.Conv2d(1024, 256, 1)
#         self.bn_fr     = nn.BatchNorm2d(n_class)
        self.score_pool3 = nn.Conv2d(256, 256, 1)
#         self.bn_score3     = nn.BatchNorm2d(n_class)
        self.score_pool4 = nn.Conv2d(256, 256, 1)
#         self.bn_score4    = nn.BatchNorm2d(n_class)
        self.upscore2 = nn.ConvTranspose2d(
            256, 256, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            256, 256, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            256, 256, 4, stride=2, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1),
#             nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
#             nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_class, 1),
#             nn.BatchNorm2d(n_class),
#             nn.ReLU(inplace=True),
        )
        self.classifier_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1)
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1)
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
        
        h = ((self.score_fr(h)))
        
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h =  ((self.score_pool4(pool4)))
        h = h[:, :, 2:2 + upscore2.size()[2], 2:2 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = ((self.score_pool3(pool3)))
        h = h[:, :,
              2:2 + upscore_pool4.size()[2],
              2:2 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 25:25 + x.size()[2], 25:25 + x.size()[3]].contiguous()
        h = self.out_conv(h)
        with torch.no_grad():
            softmax_h = torch.nn.Softmax(dim=1)(h)
            _max = torch.max(softmax_h)
            _min = torch.min(softmax_h)
            norm_result = (softmax_h-_min)/(_max-_min)
        feature = self.classifier_conv((norm_result*x).transpose(0, 1))
        n_class, _, width, height = feature.shape
        feature = nn.MaxPool2d(width, height)(feature)
        output = self.classifier(feature.view(n_class, -1)).transpose(0, 1)
        return output, h, norm_result