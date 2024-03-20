import numpy as np
import torch
#from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import copy
from torchmetrics.functional.classification import dice as DiceMetric
import torch.nn.functional as F
from evaluate import multiclass_dice_coeff, dice_coeff

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(im, lab, net, classes, patch_size=[256, 256], test_save_path=None,  case=None, z_spacing=1, device= None):
    image, label = im.cpu().detach().numpy(), lab.cpu().detach().numpy()
    #print(image.shape)
    #print(label.shape)
    #net = net.to("cpu")
    if len(image.shape) == 4:
        # print(image.shape) # (1, x,y)
        # print(label.shape)
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :, :]
            slice = slice.squeeze()
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            net=net.to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                mask_pred = F.one_hot(outputs.argmax(dim=1), classes).permute(0, 3, 1, 2).float().cpu()
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
  
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind]=pred
    
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    mask_pred = F.one_hot(torch.from_numpy(prediction), classes).permute(0, 3, 1, 2).float().cpu()
    mask_true = F.one_hot(lab, classes).permute(0, 3, 1, 2).float().cpu()
    #print(mask_pred.shape, mask_true.shape)
    dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    return dice, torch.from_numpy(prediction).cpu(), torch.from_numpy(label).cpu()