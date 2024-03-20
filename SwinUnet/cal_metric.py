import torchmetrics 
import ignite 


def jaccard(pred,label): 
    #print(pred.shape, label.shape)
    jaccard = torchmetrics.JaccardIndex(task = 'multiclass', num_classes = 3)
    return jaccard(pred,label).item()

from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from torch.nn import functional as F
# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

def  calculate_miou(pred,label): 
    default_evaluator = Engine(eval_step)
    b,h,w = pred.shape 
    pred = pred.reshape(b,h*w) 
    label = label.reshape(b,h*w) 
    pred = F.one_hot(pred, num_classes=3)
    pred = pred.permute(0,2,1) 
    cm = ConfusionMatrix(num_classes=3)
    metric = mIoU(cm, ignore_index=0)
    metric.attach(default_evaluator, 'miou')
    state = default_evaluator.run([[pred, label]])
    iou = state.metrics['miou']
    return iou

# if __name__ == '__main__': 
#     ani_pred = torch.load("pred_pt82.pth", map_location = 'cpu')
#     ani_label = torch.load("label_pt82.pth", map_location = 'cpu')

#     print("Jaccard: ", jaccard(ani_pred, ani_label))
#     print("mIoU pt: ", calculate_miou(ani_pred, ani_label)[0])
#     print("Dice pt: ", calculate_miou(ani_pred, ani_label)[1])
