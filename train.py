

import os 
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from config import opt
from collections import OrderedDict
from data.voc_data_loader import VOCDetection
from lib.ssd_loss import MultiBoxLoss
from lib.utils import detection_collate
from scheduler import GradualWarmupScheduler
from lib.multibox_encoder import MultiBoxEncoder

from lib.vgg_model import VGG_SSD
from lib.resnet_model import RESNET_SSD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_config(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            if '=' in line:
                print(line)

def adjust_learning_rate(optimizer):
    lr = 0.0001
    print('change learning rate, now learning rate is :', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':

    print_config('config.py')
    print('now runing on device : ', device)

    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)

    model = RESNET_SSD(opt.backbone_network_name, 21, [4,6,6,6,4,4])
    
    if opt.resume:
        print('loading checkpoint...')
        if opt.Data_Parallel == False:
            state_dict = torch.load(opt.resume)
            model.load_state_dict(state_dict)

        elif opt.Data_Parallel == True:
            state_dict = torch.load(opt.resume)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    else:
        pretrained_weights = torch.load(opt.save_folder + opt.basenet)
        print('Loading base network...')
        
        # If you use vgg
        # model.VGG.load_state_dict(pretrained_weights)

        # If you use resnet
        model.ResNet.load_state_dict(pretrained_weights)

    model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.train()

    mb = MultiBoxEncoder(opt)
        
    image_sets = [['2007', 'trainval'], ['2012', 'trainval']]
    dataset = VOCDetection(opt, image_sets=image_sets, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, collate_fn=detection_collate, num_workers=4)

    criterion = MultiBoxLoss(opt.num_classes, opt.neg_radio).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    
    if opt.lr_scheduler == True:
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.25, total_epoch=2, after_scheduler=cosine_scheduler)

    try:
        if not(os.path.isdir(os.path.join(opt.save_folder,opt.exp_information_name))):
            os.makedirs(os.path.join(opt.save_folder,opt.exp_information_name))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!!!!")
            raise

                    
    f=open(os.path.join(opt.save_folder,opt.exp_information_name,opt.exp_information_name +'.txt'),'w')
    f.write('backbone_network_name = ' + opt.backbone_network_name +'\nexp_information_name = ' + opt.exp_information_name +\
            '\n\n[hyperparameter]' + \
            '\ntotal_epoch = ' + str(opt.epoch) + \
            '\nbatch_size = ' + str(opt.batch_size) + \
            '\nlearning rate = ' + str(opt.lr) + \
            '\nlr_scheduler = ' + str(opt.lr_scheduler) + \
            '\nweight_decay = ' + str(opt.weight_decay) + \
            '\nfeature_maps = ' + str(opt.grids) + \
            '\naspect_ratios = ' + str(opt.aspect_ratios) +\
            '\nsteps = ' + str(opt.steps) + \
            '\nmin_dim = ' + str(opt.min_size) + 
            '\nanchor_num = ' + str(opt.anchor_num) + \
            '\nmeans = ' + str(opt.mean) + \
            '\nvariance = ' + str(opt.variance) + \
            '\n\n\n\n\n')
    f.close()
    
    print('start training........')
    for e in range(opt.epoch+1):
        if e % opt.lr_reduce_epoch == 0:
            adjust_learning_rate(optimizer)
            
        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0        
        total_time = 0.0
        time_tool = 0
        
        for i , (img, boxes) in enumerate(dataloader):
            img = img.to(device)
            gt_boxes = []
            gt_labels = []
            for box in boxes:
                labels = box[:, 4]
                box = box[:, :-1]
                match_loc, match_label = mb.encode(box, labels)
            
                gt_boxes.append(match_loc)
                gt_labels.append(match_label)
            
            gt_boxes = torch.FloatTensor(gt_boxes).to(device)
            gt_labels = torch.LongTensor(gt_labels).to(device)
            
            if time_tool == 0:
                start = time.time()
                time_tool = 1

            p_loc, p_label = model(img)

            loc_loss, cls_loss = criterion(p_loc, p_label, gt_boxes, gt_labels)

            loss = loc_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if e >= opt.lr_scheduler_start_epoch: 
                scheduler.step()
                
            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()

            if i % opt.log_fn == 0:
                avg_loc = total_loc_loss / (i+1)
                avg_cls = total_cls_loss / (i+1)
                avg_loss = total_loss / (i+1)
                
                end = time.time()
                moment_time = end-start
                total_time += moment_time
                m,s = divmod(total_time,60)
                time_tool = 0
                
                f=open(os.path.join(opt.save_folder,opt.exp_information_name,opt.exp_information_name +'.txt'),'a')
                f.write('\nepoch [{}] | lr [{:.10f}] | batch_idx [{}] | loc_loss [{:.3f}] | cls_loss [{:.3f}] | total_loss [{:.3f}] | [{:.0f}min,{:.0f}sec]'.format(e,optimizer.param_groups[0]['lr'] , i, avg_loc, avg_cls, total_loss, m,s))
                f.close()

                print('epoch [{}] | lr [{:.10f}] | batch_idx [{}] | loc_loss [{:.3f}] | cls_loss [{:.3f}] | total_loss [{:.3f}] | [{:.0f}min,{:.0f}sec]'.format(e,optimizer.param_groups[0]['lr'] , i, avg_loc, avg_cls, total_loss, m,s))

        if e % 5 ==0:
            print('Saving state, epoch:', e)
            torch.save(model.state_dict(), os.path.join(opt.save_folder,opt.exp_information_name,'locloss_{:.3f}_clsloss_{:.3f}_total_loss_{:.3f}.pth'.format(avg_loc,avg_cls,total_loss)))

