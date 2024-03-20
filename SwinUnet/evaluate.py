import torch
import torch.nn.functional as F
from tqdm import tqdm
from cal_metric import jaccard, calculate_miou
import torch
from torch import Tensor


def  dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def count_f(pred: Tensor, true: Tensor):
    pred = pred.squeeze() 
    true = true.squeeze()
    #print(pred.shape, true.shape)
    inter = pred*true 
    union = pred+true - inter 
    inter_count = torch.sum(inter, dim = (1,2))
    union_count = torch.sum(union, dim = (1,2)) 
    return inter_count, union_count

@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    list_pred= []
    list_label = []
    # iterate over the validation set
    with torch.autocast(device if device != 'mps' else 'cpu'):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['label']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.num_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.num_classes, 'True mask indices should be in [0, num_classes['
                # convert to one-hot format
                tmp_true = mask_true.squeeze()
                tmp_pred = mask_pred.argmax(dim=1).squeeze()
                list_pred.append(tmp_true.cpu())
                list_label.append(tmp_pred.cpu())
                mask_true = F.one_hot(mask_true, net.num_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                

                


    net.train()
    list_pred = torch.stack(list_pred, dim=0)
    list_label = torch.stack(list_label, dim=0)

    j = jaccard(list_pred, list_label)
    miou = calculate_miou(list_pred, list_label)
    return dice_score / max(num_val_batches, 1), j, miou