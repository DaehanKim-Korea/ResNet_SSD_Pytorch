# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

extract = {
        'resnet_18': {'b':[7,9],'e':[4,6,8,10]}
        'resnet_34': {'b':[10,16],'e':[4,6,8,10]}
        'resnet_34': {'b':[10,16],'e':[4,6,8,10]}
        'resnet_101': {'b':[10,33],'e':[4,6,8,10]}
    }
    
class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)
        

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x
        
def make_layers(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = []
    bbl=block(inplanes, planes, stride, downsample)
    bbl.out_channels=planes*block.expansion
    layers.append(bbl)
    
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        bbl=block(inplanes, planes)
        bbl.out_channels=planes*block.expansion
        layers.append(bbl)
    return layers

def ResNet(model_mode):
    cfg = [3,64,128,256,512]
    resnet = []
    resnet += [nn.Conv2d(cfg[0], cfg[1], kernel_size=7, stride=2, padding=3,bias=False),
               nn.BatchNorm2d(cfg[1]),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

    if model_mode == 'resnet_18'
        num_block = [2,2,2,2]
        block= torchvision.models.resnet.BasicBlock
        resnet += make_layers(block,cfg[1],cfg[1],num_block[0],stride=1)                 # 75 x 75 
        resnet += make_layers(block,cfg[1]*block.expansion,cfg[2],num_block[1],stride=2) # 38 x 38
        resnet += make_layers(block,cfg[2]*block.expansion,cfg[3],num_block[2],stride=2) # 19 x 19
        resnet += make_layers(block,cfg[3]*block.expansion,cfg[4],num_block[3],stride=2) # 10 x 10
    if model_mode == 'resnet_34'
        num_block = [3,4,6,3]
        block= torchvision.models.resnet.BasicBlock
        resnet += make_layers(block,cfg[1],cfg[1],num_block[0],stride=1)                 # 75 x 75 
        resnet += make_layers(block,cfg[1]*block.expansion,cfg[2],num_block[1],stride=2) # 38 x 38
        resnet += make_layers(block,cfg[2]*block.expansion,cfg[3],num_block[2],stride=2) # 19 x 19
        resnet += make_layers(block,cfg[3]*block.expansion,cfg[4],num_block[3],stride=2) # 10 x 10
    if model_mode == 'resnet_50'
        num_block = [3,4,6,3]
        block= torchvision.models.resnet.Bottleneck
        resnet += make_layers(block,cfg[1],cfg[1],num_block[0],stride=1)                 # 75 x 75 
        resnet += make_layers(block,cfg[1]*block.expansion,cfg[2],num_block[1],stride=2) # 38 x 38
        resnet += make_layers(block,cfg[2]*block.expansion,cfg[3],num_block[2],stride=2) # 19 x 19
        resnet += make_layers(block,cfg[3]*block.expansion,cfg[4],num_block[3],stride=2) # 10 x 10
    if model_mode == 'resnet_101'
        num_block = [3,4,23,3]
        block= torchvision.models.resnet.Bottleneck
        resnet += make_layers(block,cfg[1],cfg[1],num_block[0],stride=1)                 # 75 x 75 
        resnet += make_layers(block,cfg[1]*block.expansion,cfg[2],num_block[1],stride=2) # 38 x 38
        resnet += make_layers(block,cfg[2]*block.expansion,cfg[3],num_block[2],stride=2) # 19 x 19
        resnet += make_layers(block,cfg[3]*block.expansion,cfg[4],num_block[3],stride=2) # 10 x 10
            
    return resnet
    
def ResNet_Extra(model_mode):
    cfg = [128,256,512,1024,2048]
    layers = []
    
    if model_mode == 'resnet_18'
        extra_pool_01 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        extra_conv_01 = nn.Conv2d(cfg[2], cfg[1], kernel_size=3, padding=6, dilation=6)
        extra_conv_02 = nn.Conv2d(cfg[1], cfg[3], kernel_size=1)
        # 10 x 10

        extra_conv_03 = nn.Conv2d(cfg[3],cfg[1],kernel_size=1, stride=1)
        extra_conv_04 = nn.Conv2d(cfg[1],cfg[2],kernel_size=3, stride=1, padding=1) 
        # 10 x 10

        extra_conv_05 = nn.Conv2d(cfg[2],cfg[0],kernel_size=1, stride=1) 
        extra_conv_06 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=2, padding=1) 
        # 5 x 5

        extra_conv_07 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_08 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)
        # 5 x 5

        extra_conv_09 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_10 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)
        # 1 x 1

        layers = [extra_pool_01, extra_conv_01,extra_conv_02,extra_conv_03, extra_conv_04, extra_conv_05,extra_conv_06, extra_conv_07, extra_conv_08, extra_conv_09, extra_conv_10]
        
    elif model_mode == 'resnet_34'

        extra_pool_01 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        extra_conv_01 = nn.Conv2d(cfg[2], cfg[1], kernel_size=3, padding=6, dilation=6)
        extra_conv_02 = nn.Conv2d(cfg[1], cfg[3], kernel_size=1)
        # 10 x 10

        extra_conv_03 = nn.Conv2d(cfg[3],cfg[1],kernel_size=1, stride=1)
        extra_conv_04 = nn.Conv2d(cfg[1],cfg[2],kernel_size=3, stride=1, padding=1) 
        # 10 x 10

        extra_conv_05 = nn.Conv2d(cfg[2],cfg[0],kernel_size=1, stride=1) 
        extra_conv_06 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=2, padding=1) 
        # 5 x 5

        extra_conv_07 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_08 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)
        # 5 x 5

        extra_conv_09 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_10 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)

        layers = [extra_pool_01, extra_conv_01,extra_conv_02,extra_conv_03, extra_conv_04, extra_conv_05,extra_conv_06, extra_conv_07, extra_conv_08, extra_conv_09, extra_conv_10]
    elif model_mode == 'resnet_50'
        extra_pool_01 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        extra_conv_01 = nn.Conv2d(cfg[4], cfg[2], kernel_size=3, padding=6, dilation=6)
        extra_conv_02 = nn.Conv2d(cfg[2], cfg[3], kernel_size=1)
        # 10 x 10

        extra_conv_03 = nn.Conv2d(cfg[3],cfg[1],kernel_size=1, stride=1)
        extra_conv_04 = nn.Conv2d(cfg[1],cfg[2],kernel_size=3, stride=1, padding=1) 
        # 10 x 10

        extra_conv_05 = nn.Conv2d(cfg[2],cfg[0],kernel_size=1, stride=1) 
        extra_conv_06 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=2, padding=1) 
        # 5 x 5

        extra_conv_07 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_08 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)
        # 5 x 5

        extra_conv_09 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_10 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)
        # 1 x 1

        layers = [extra_pool_01, extra_conv_01,extra_conv_02,extra_conv_03, extra_conv_04, extra_conv_05,extra_conv_06, extra_conv_07, extra_conv_08, extra_conv_09, extra_conv_10]
    elif model_mode == 'resnet_101'
        extra_pool_01 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        extra_conv_01 = nn.Conv2d(cfg[4], cfg[2], kernel_size=3, padding=6, dilation=6)
        extra_conv_02 = nn.Conv2d(cfg[2], cfg[3], kernel_size=1)
        # 10 x 10

        extra_conv_03 = nn.Conv2d(cfg[3],cfg[1],kernel_size=1, stride=1)
        extra_conv_04 = nn.Conv2d(cfg[1],cfg[2],kernel_size=3, stride=1, padding=1) 
        # 10 x 10

        extra_conv_05 = nn.Conv2d(cfg[2],cfg[0],kernel_size=1, stride=1) 
        extra_conv_06 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=2, padding=1) 
        # 5 x 5

        extra_conv_07 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_08 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)
        # 5 x 5

        extra_conv_09 = nn.Conv2d(cfg[1],cfg[0],kernel_size=1, stride=1)
        extra_conv_10 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1)
        # 1 x 1

        layers = [extra_pool_01, extra_conv_01,extra_conv_02,extra_conv_03, extra_conv_04, extra_conv_05,extra_conv_06, extra_conv_07, extra_conv_08, extra_conv_09, extra_conv_10]

    return layers

def Feature_extractor(model, model_list, extral, bboxes, num_classes):
    
    loc_layers = []
    conf_layers = []
    
    k=0
    for v in extract[model]['b']:
        loc_layers += [nn.Conv2d(model_list[v].out_channels,
                        bboxes[k]* 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(model_list[v].out_channels,
                        bboxes[k]*num_classes, kernel_size=3, padding=1)]
        k+=1
    
    for v in extract[model]['e']:
        loc_layers += [nn.Conv2d(extral[v].out_channels,
                        bboxes[k]* 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extral[v].out_channels,
                        bboxes[k]*num_classes, kernel_size=3, padding=1)]
        k+=1
        
        
    return loc_layers, conf_layers

class RESNET_SSD(nn.Module):

    def __init__(self, model, num_classes, bboxes):
        super(RESNET_SSD, self).__init__()

        self.num_classes = num_classes
        self.bboxes = bboxes
        self.model = model

        if self.model = 'resnet_18':
            self.L2Norm = L2Norm(128, 20)
            self.ResNet_list = ResNet(self.model)
            self.extra_list = ResNet_Extra(self.model)
            self.loc_layers_list, self.conf_layers_list = Feature_extractor(self.model,self.ResNet_list, self.extra_list, self.bboxes, self.num_classes)

            self.ResNet = nn.ModuleList(self.ResNet_list)
            self.extras = nn.ModuleList(self.extra_list)
            self.loc = nn.ModuleList(self.loc_layers_list)
            self.conf = nn.ModuleList(self.conf_layers_list)
            
        elif self.model = 'resnet_34':
            self.L2Norm = L2Norm(128, 20)
            self.ResNet_list = ResNet(self.model)
            self.extra_list = ResNet_Extra(self.model)
            self.loc_layers_list, self.conf_layers_list = Feature_extractor(self.model,self.ResNet_list, self.extra_list, self.bboxes, self.num_classes)

            self.ResNet = nn.ModuleList(self.ResNet_list)
            self.extras = nn.ModuleList(self.extra_list)
            self.loc = nn.ModuleList(self.loc_layers_list)
            self.conf = nn.ModuleList(self.conf_layers_list)

        elif self.model = 'resnet_50':
            self.L2Norm = L2Norm(512, 20)
            self.ResNet_list = ResNet(self.model)
            self.extra_list = ResNet_Extra(self.model)
            self.loc_layers_list, self.conf_layers_list = Feature_extractor(self.model,self.ResNet_list, self.extra_list, self.bboxes, self.num_classes)

            self.ResNet = nn.ModuleList(self.ResNet_list)
            self.extras = nn.ModuleList(self.extra_list)
            self.loc = nn.ModuleList(self.loc_layers_list)
            self.conf = nn.ModuleList(self.conf_layers_list)

        elif self.model = 'resnet_101':
            self.L2Norm = L2Norm(512, 20)
            self.ResNet_list = ResNet(self.model)
            self.extra_list = ResNet_Extra(self.model)
            self.loc_layers_list, self.conf_layers_list = Feature_extractor(self.model,self.ResNet_list, self.extra_list, self.bboxes, self.num_classes)

            self.ResNet = nn.ModuleList(self.ResNet_list)
            self.extras = nn.ModuleList(self.extra_list)
            self.loc = nn.ModuleList(self.loc_layers_list)
            self.conf = nn.ModuleList(self.conf_layers_list)

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        if self.model=='resnet_18'
            for k, v in enumerate(self.ResNet):
                x = v(x)
                if k==7 or k==9:
                    if k==7:
                        sources.append(self.L2Norm(x))               
                    else:
                        sources.append(x)
                            
            for i, v in enumerate(self.extras):
                if i==0:
                    x = v(x)
                if i!=0:
                    x = F.relu(v(x), inplace=True)
                if i==4 or i==6 or i==8 or i==10:
                    sources.append(x)

        elif self.model=='resnet_34'
            for k, v in enumerate(self.ResNet):
                x = v(x)
                if k==10 or k==16:
                    if k==10:
                        sources.append(self.L2Norm(x))               
                    else:
                        sources.append(x)
                            
            for i, v in enumerate(self.extras):
                if i==0:
                    x = v(x)
                if i!=0:
                    x = F.relu(v(x), inplace=True)
                if i==4 or i==6 or i==8 or i==10:
                    sources.append(x)

        elif self.model=='resnet_50'
            for k, v in enumerate(self.ResNet):
                x = v(x)
                if k==10 or k==16:
                    if k==10:
                        sources.append(self.L2Norm(x))               
                    else:
                        sources.append(x)
                            
            for i, v in enumerate(self.extras):
                if i==0:
                    x = v(x)
                if i!=0:
                    x = F.relu(v(x), inplace=True)
                if i==4 or i==6 or i==8 or i==10:
                    sources.append(x)

        elif self.model=='resnet_101'
            for k, v in enumerate(self.ResNet):
                x = v(x)
                if k==10 or k==33:
                    if k==7:
                        sources.append(self.L2Norm(x))               
                    else:
                        sources.append(x)
                            
            for i, v in enumerate(self.extras):
                if i==0:
                    x = v(x)
                if i!=0:
                    x = F.relu(v(x), inplace=True)
                if i==4 or i==6 or i==8 or i==10:
                    sources.append(x)
                        
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        
        return loc, conf

