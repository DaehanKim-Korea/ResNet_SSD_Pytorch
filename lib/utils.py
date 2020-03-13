import cv2
import torch
import numpy as np

def point_form(boxes):
    '''
    Convert [cx, cy, w, h] types to [xmin, ymin, xmax, ymax] forms    
    '''

    tl = boxes[:, :2] - boxes[:, 2:]/2
    br = boxes[:, :2] + boxes[:, 2:]/2

    return np.concatenate([tl, br], axis=1)


def detection_collate(batch):
    '''
    Since the gt number of each sample is not necessarily the same, we will define the splice function ourselves    '''
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs), np.array(targets)



def bbox_iou(box_a, box_b):
    '''
    Calculate the iOU of two Box Groups    box_a : (m, 4)
    box_b : (n, 4)
    '''
    m = box_a.shape[0]
    n = box_b.shape[0]

    # Broadcasting, doing the equivalent of (m, 1, 2) and (1, n, 2) operations and finally getting (m, n, 2) size    
    tl = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    br = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])

    wh = np.maximum(br-tl, 0)
    
    inner = wh[:, :, 0]*wh[:, :, 1]

    a = box_a[:, 2:] - box_a[:, :2]
    b = box_b[:, 2:] - box_b[:, :2]

    a = a[:, 0] * a[:, 1]
    b = b[:, 0] * b[:, 1]

    a = a[:, None]
    b = b[None, :]

    # Last but not least. (m,n) / (m, 1) + (1,n) - (m,n)
    # get a matrix (m,n) in which each point (i,j) represents the iou of i and j

    return inner / (a+b-inner)


def nms(boxes, score, threshold=0.4):
    '''
    Nms operation via iou
    boxes : (n, 4)
    score: (n, )
    '''

    sort_ids = np.argsort(score)
    pick = []
    while len(sort_ids) > 0:
        i = sort_ids[-1]
        pick.append(i)
        if len(sort_ids) == 1:
            break

        sort_ids = sort_ids[:-1]
        box = boxes[i].reshape(1, 4)
        ious = bbox_iou(box, boxes[sort_ids]).reshape(-1)

        sort_ids = np.delete(sort_ids, np.where(ious > threshold)[0])

    return pick




def detect(locations, scores, nms_threshold, gt_threshold):
    '''
    locations : postdecode coordinates (num_anchors, 4)
    scores : a predicted score (num_anchors, 21)
    nms_threshold : Valuation of nms
    gt_threshold : Thresholds considered to be true object GROUND TRUTH
    '''

    scores = scores[:, 1:] #Category 0 is background, filtered out

    keep_boxes = []
    keep_confs = []
    keep_labels = []
    
    for i in range(scores.shape[1]):
        mask = scores[:, i] >= gt_threshold
        label_scores = scores[mask, i] 
        label_boxes = locations[mask]
        if len(label_scores) == 0:
            continue

        pick = nms(label_boxes, label_scores, threshold=nms_threshold)
        label_scores = label_scores[pick]
        label_boxes = label_boxes[pick]
        

        keep_boxes.append(label_boxes.reshape(-1))
        keep_confs.append(label_scores)
        keep_labels.extend([i]*len(label_scores))
    
    if len(keep_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
        
    
    keep_boxes = np.concatenate(keep_boxes, axis=0).reshape(-1, 4)

    keep_confs = np.concatenate(keep_confs, axis=0)
    keep_labels = np.array(keep_labels).reshape(-1)
#     print(keep_boxes.shape)
#     print(keep_confs.shape)
#     print(keep_labels.shape)

    return keep_boxes, keep_confs, keep_labels

def draw_rectangle(src_img, labels, conf, locations, label_map):
    '''
    src_img : a picture to be framed
    labels : Object gets label, digital form
    conf : the probability that there is an object there
    locations : Coordinates
    label_map : Map Labels back to category name
    
    return
        draw a picture of a frame
    '''
    num_obj = len(labels)
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    img = src_img.copy()
    for i in range(num_obj):
        tl = tuple(locations[i][:2])
        br = tuple(locations[i][2:])
        
        cv2.rectangle(img,
                      tl,
                      br,
                      COLORS[i%3], 3)
        cv2.putText(img, label_map[labels[i]], tl,
                    FONT, 1, (255, 255, 255), 2)
    
    img = img[:, :, ::-1]

    return img