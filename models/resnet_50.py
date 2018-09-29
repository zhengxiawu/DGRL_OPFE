import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet50': '/home/zhengxiawu/.torch/models/resnet50-19c8e357.pth',
}
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion * 2, kwargs['num_class'])
        #normalze the weight with
        self.is_train = bool(kwargs['is_train'])
        self.saliency = str(kwargs['saliency'])
        self.pool_type = str(kwargs['pool_type'])
        self.scale = int(kwargs['scale'])
        self.threshold = float(kwargs['threshold']) if kwargs.has_key('threshold') else 'none'
        self.phase = str(kwargs['phase']) if kwargs.has_key('phase') else 'none'


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    def forward(self, x):
        #**kwargs form
        #pool_type:default max_avg, can be set max, avg
        #scale: int default 128
        #scda:True or false, if use saliency
        #is_train
        x = self.extract_conv_feature(x)
        if self.saliency=='scda':
            scda_x = torch.sum(x,1,keepdim=True)
            mean_x = torch.mean(scda_x.view(scda_x.size(0),-1),1,True)
            scda_x = scda_x - mean_x
            scda_x = scda_x>0
            scda_x = scda_x.float()
            x = x * scda_x
        elif self.saliency == 'oc_mask':
            object_tive_ness = torch.sum(x,1,keepdim=True)
            max_object_score = 2* torch.mean(object_tive_ness.view(object_tive_ness.size(0),-1),1,True)
            object_tive_ness = object_tive_ness / max_object_score
            _,_, size_w, size_h = object_tive_ness.shape
            prior_feature = np.indices((size_w,size_h))
            prior_feature = prior_feature + 1
            prior_feature = np.transpose(prior_feature, axes=(1, 2, 0))
            prior_feature = prior_feature - (np.array((size_w,size_h)) / 2.)
            sigma = size_h if size_h < size_w else size_w
            sigma = sigma / 3.
            prior_feature = np.exp(-1 * np.sum(prior_feature ** 2, axis=2) / (2 * (sigma ** 2)))
            prior_feature = np.reshape(prior_feature,(1,1,size_w,size_h))
            prior_feature_tensor = torch.Tensor(prior_feature).cuda()
            indicate_mat = object_tive_ness + prior_feature_tensor> self.threshold
            indicate_mat = indicate_mat.float()
            x = x * indicate_mat
        if self.phase == 'extract_conv_feature':
            return x
        if self.pool_type == 'max_avg':
            avg_x = self.global_avg_pool(x)
            avg_x = avg_x.view(avg_x.size(0), -1)
            avg_x = F.normalize(avg_x,p=2,dim=1)
            max_x = self.global_max_pool(x)
            max_x = max_x.view(max_x.size(0), -1)
            max_x = F.normalize(max_x,p=2,dim=1)
            x = torch.cat((avg_x,max_x),dim=1)
        x = x * self.scale
        # the last fc layer can be treat as distance compute
        if self.is_train:
            fc_weight_relu = self.relu(self.class_fc.weight)
            self.class_fc.weight.data = fc_weight_relu
            x = self.class_fc(x)
            return x,fc_weight_relu
        return x

def resnet_50(pretrained=False,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = torch.load(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model