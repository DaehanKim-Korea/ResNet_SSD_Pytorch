import os 
import cv2
import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from lib.utils import nms
from lib.utils import detect
from lib.voc_eval import voc_eval
from collections import OrderedDict
from lib.ssd_loss import MultiBoxLoss
from lib.resnet_model import RESNET_SSD
from lib.utils import detection_collate
from data.voc_eval_dataloader import VOC_LABELS
from data.voc_eval_dataloader import VOCDetection
from lib.augmentations import preproc_for_test
from lib.multibox_encoder import MultiBoxEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    default='weights/pre_resnet_101_03/locloss_0.651_clsloss_1.765_total_loss_627.728.pth',
                    type=str,
                    help='model checkpoint used to eval VOC dataset')

parser.add_argument('--save_folder',
                    default='result',
                    type=str,
                    help='eval result save folder')
args = parser.parse_args()

output_dir = args.save_folder
checkpoint = args.model

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

voc_root = '/jupyter/VOCdevkit' #   '/data/PASCAL'
annopath = os.path.join(voc_root, 'VOC2007', 'Annotations', "%s.xml")   
imgpath = os.path.join(voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')    
imgsetpath = os.path.join(voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')   
cachedir = os.path.join( os.getcwd(), 'annotations_cache')    

if __name__ == '__main__': 
    
    print('using {} to eval, use cpu may take an hour to complete !!'.format(device))
    # If you use vgg
    # model.VGG.load_state_dict(pretrained_weights)

    # If you use resnet
    model.ResNet.load_state_dict(pretrained_weights)

    print('loading checkpoint from {}'.format(checkpoint))
    state_dict = torch.load(checkpoint, map_location=None if torch.cuda.is_available() else 'cpu')
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
                            
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    print('model loaded')

    multibox_encoder = MultiBoxEncoder(opt)
    image_sets = [['2007', 'test']]
    test_dataset = VOCDetection(opt, image_sets=image_sets, is_train=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = [
        open(
            os.path.join(
                output_dir, '{:s}.txt'.format(label)),
            mode='w')
        for label in VOC_LABELS]

    print('start detect.........')

    for i in tqdm.tqdm(range(len(test_dataset))):
        src = test_dataset[i][0]
        
        img_name = os.path.basename(test_dataset.ids[i][0]).split('.')[0]
        image = preproc_for_test(src, opt.min_size, opt.mean)
        image = torch.from_numpy(image).cuda()
        with torch.no_grad():
            loc, conf = model(image.unsqueeze(0))
        loc = loc[0]
        conf = conf[0]
        conf = F.softmax(conf, dim=1)
        conf = conf.cpu().numpy()
        loc = loc.cpu().numpy()

        decode_loc = multibox_encoder.decode(loc)
        gt_boxes, gt_confs, gt_labels = detect(decode_loc, conf, nms_threshold=0.5, gt_threshold=0.01)

        #no object detected
        if len(gt_boxes) == 0:
            continue

        h, w = src.shape[:2]
        gt_boxes[:, 0] = gt_boxes[:, 0] * w
        gt_boxes[:, 1] = gt_boxes[:, 1] * h
        gt_boxes[:, 2] = gt_boxes[:, 2] * w
        gt_boxes[:, 3] = gt_boxes[:, 3] * h


        for box, label, score in zip(gt_boxes, gt_labels, gt_confs):
            print(img_name, "{:.3f}".format(score), "{:.1f} {:.1f} {:.1f} {:.1f}".format(*box), file=files[label])


    for f in files:
        f.close()
    
    print(imgsetpath.format('test'))
    print('start cal MAP.........')
    aps = []
    for f in os.listdir(output_dir):
        filename = os.path.join(output_dir, f)
        class_name = f.split('.')[0]
        rec, prec, ap = voc_eval(filename, annopath, imgsetpath.format('test'), class_name, cachedir, ovthresh=0.1, use_07_metric=True)
        print(class_name, ap)
        aps.append(ap)

    print('mean MAP is : ', np.mean(aps))
    


