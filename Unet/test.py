import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from cal_metric import jaccard,dice, calculate_miou
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import MedDataset
from utils.dice_score import dice_loss, multiclass_dice_coeff, count_f
from evaluate import evaluate




def test_model(model, image, device): 
    image = image.to(device)
    model.eval()  
    out = model(image) 
    return out


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--dataset_domain', type=str, default="animal", help='Choose one of the following: "phantom", "animal", "sim", "real"')
    parser.add_argument('--ckpt_dir', type=str, default = './pt_ani_lr-1e-05_seed0/best_val.pth',  help='checkpoint directory to load') # you can change the checkpoint load here

    return parser.parse_args()

def main():
    args = get_args()

    #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain = args.dataset_domain

    # Change here to adapt to your data
    pt_test_dir_img = Path(f'../data/{domain}_test/images/')
    pt_test_dir_mask = Path(f'../data/{domain}_test/masks/')

    
    # n_channels=3 for RGB images
    assert domain in ["animal", "phantom", "sim", "real"], f'The domain of dataset ({domain}) must be one of the following:["animal", "phantom", "sim", "real"] '
    # different domains raise different input and output
    if domain == "animal" or domain == "phantom": 
        n_channels = 3
        args.classes = 3 
    elif domain == 'sim': 
        n_channels = 3
        args.classes = 2
    else: 
        n_channels = 1 
        args.classes = 2


    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    
    dir_checkpoint = Path(args.ckpt_dir)
    ckpt = torch.load(dir_checkpoint)
    #del ckpt['mask_values']
    model.load_state_dict(ckpt)

    model.to(device=device)
    val_set = MedDataset(images_dir=pt_test_dir_img, mask_dir=pt_test_dir_mask,domain='animal')
    val_loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)
    

    acc , val_score, jac, miou = evaluate(model, val_loader, device, amp=True)
    print(f'Accuracy: {acc} -- Dice score: {val_score} -- Jaccard: {jac} -- mIoU: {miou} ')

if __name__ == "__main__":
    main()

