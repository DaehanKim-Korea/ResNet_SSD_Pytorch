import torch
import argparse
import numpy as np
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

parser = argparse.ArgumentParser()
parser.add_argument('--choice', 
                    default='resnet18',
                    type=str,
                    help='model checkpoint used to eval VOC dataset')

parser.add_argument('--save_folder',
                    default='result',
                    type=str,
                    help='eval result save folder')
args = parser.parse_args()


if args.chice == 'resnet18'
    r18=model_zoo.load_url(model_urls['resnet18'])
    rr18=OrderedDict()

    nk0={'conv1':'0','bn1':'1'}

    bb=np.array([2,2,2,2])
    #old_key=
    for i,_ in enumerate(range(len(r18))):
        k, v = r18.popitem(False)
        #print(i,k)
        key=k.split('.')
        nk=''
        if key[0] in nk0.keys():
            key[0]=nk0[key[0]]
            nk=".".join(key)
            r18[nk]=v
        elif key[0].startswith('layer'):
            layer=int(key[0][-1])-1
            key[0]= str(bb[:layer].sum()+int(key[1])+4)
            del key[1]
            nk=".".join(key)
            r18[nk]=v
        print(i,nk,k)    
        
    torch.save(r18, 'resnet18.pth') 

elif args.choice == 'resnet34'

    r34=model_zoo.load_url(model_urls['resnet34'])
    rr34=OrderedDict()

    nk0={'conv1':'0','bn1':'1'}

    bb=np.array([3,4,6,3])
    #old_key=
    for i,_ in enumerate(range(len(r34))):
        k, v = r34.popitem(False)
        #print(i,k)
        key=k.split('.')
        nk=''
        if key[0] in nk0.keys():
            key[0]=nk0[key[0]]
            nk=".".join(key)
            r34[nk]=v
        elif key[0].startswith('layer'):
            layer=int(key[0][-1])-1
            key[0]= str(bb[:layer].sum()+int(key[1])+4)
            del key[1]
            nk=".".join(key)
            r34[nk]=v
        print(i,nk,k)    
        
    torch.save(r34, 'resnet34.pth') 

elif args.chice == 'resnet50'

    r50=model_zoo.load_url(model_urls['resnet50'])
    rr50=OrderedDict()

    nk0={'conv1':'0','bn1':'1'}

    bb=np.array([3,4,6,3])
    #old_key=
    for i,_ in enumerate(range(len(r50))):
        k, v = r50.popitem(False)
        #print(i,k)
        key=k.split('.')
        nk=''
        if key[0] in nk0.keys():
            key[0]=nk0[key[0]]
            nk=".".join(key)
            r50[nk]=v
        elif key[0].startswith('layer'):
            layer=int(key[0][-1])-1
            key[0]= str(bb[:layer].sum()+int(key[1])+4)
            del key[1]
            nk=".".join(key)
            r50[nk]=v
        print(i,nk,k)    
        
    torch.save(r50, 'resnet50.pth') 

elif args.chice == 'resnet101'

    r101=model_zoo.load_url(model_urls['resnet101'])
    rr101=OrderedDict()

    nk0={'conv1':'0','bn1':'1'}

    bb=np.array([3,4,23,3])
    #old_key=
    for i,_ in enumerate(range(len(r101))):
        k, v = r101.popitem(False)
        #print(i,k)
        key=k.split('.')
        nk=''
        if key[0] in nk0.keys():
            key[0]=nk0[key[0]]
            nk=".".join(key)
            r101[nk]=v
        elif key[0].startswith('layer'):
            layer=int(key[0][-1])-1
            key[0]= str(bb[:layer].sum()+int(key[1])+4)
            del key[1]
            nk=".".join(key)
            r101[nk]=v
        print(i,nk,k)    
        
    torch.save(r101, 'resnet101.pth') 