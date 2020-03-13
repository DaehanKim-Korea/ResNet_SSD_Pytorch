

class Config:

	# insert your VOCdevkit path
    VOC_TRAIN_ROOT = './data/VOCdataset/VOC_train_val/VOCdevkit'
    VOC_TEST_ROOT = './data/VOCdataset/VOC_test/VOCdevkit'

    num_classes = 21
    
    resume = None
    Data_Parallel = False
    
    # if 
    # 'weights/pre_resnet_101_02/locloss_0.661_clsloss_1.801_total_loss_1279.111.pth'
    
    lr_scheduler = True
    lr = 0.001
    gamma = 0.2
    momentum = 0.9
    weight_decay = 5e-4

    batch_size = 64
    epoch = 120

    lr_reduce_epoch = 100
    lr_scheduler = 'warmup_cosine'
    lr_scheduler_start_epoch = 60
    
    save_folder = 'weights/'
    backbone_network_name = 'resnet_18'
    exp_information_name = 'dummy_exp'
    basenet = 'torchvision_pretrained_resnet_rmfc/resnet18.pth'

    log_fn = 10 
    neg_radio = 3
   
    min_size = 300
    grids = (38, 19, 10, 5, 3, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)] 
    anchor_num = [4, 6, 6, 6, 4, 4]

    mean = (104, 117, 123)
    variance = (0.1, 0.2)

opt = Config()
