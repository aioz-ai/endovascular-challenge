import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.med_dataset import MedDataset, RandomGenerator
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from cal_metric import calculate_miou, jaccard


device = "cuda"
parser = argparse.ArgumentParser()


parser.add_argument('--dataset_domain', type=str, 
                        default="animal", help='Choose one of the following: "phantom", "animal", "sim", "real"')


parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.02, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=40000, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--ckpt_dir', type=str, default = '',  help='checkpoint directory to load') # you can change the checkpoint load here
args = parser.parse_args()


def inference(args, model, test_save_path=None):


    #db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    db_test = MedDataset(images_dir=f'../data/{args.domain}_test/images', mask_dir=f'../data/{args.domain}_test/masks', transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]), domain = args.domain)\
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    dice_score = 0.0
    accuracy = 0.0
    list_pred = [] 
    list_label =[] 
    for test_sample in tqdm(testloader): 
        image, label = test_sample["image"], test_sample["label"]
        acc, dice_per_sample,pred,lab= test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                test_save_path=None,case=None, z_spacing=1, device = device)
        dice_score += dice_per_sample
        accuracy+=acc
        list_pred.append(pred)
        list_label.append(lab) 

    list_pred = torch.stack(list_pred, dim=0).squeeze()
    list_label = torch.stack(list_label, dim = 0).squeeze()

    jacc = jaccard(list_pred, list_label)
    miou = calculate_miou(list_pred, list_label)
    dice_score = dice_score / len(db_test)
    accuracy =accuracy/len(db_test)
    print(f"Accuracy: {accuracy} -- Jaccard: {jacc} -- mIoU: {miou} -- Dice score: {dice_score}")




if __name__ == "__main__":
    

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            #'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 3,
            'z_spacing': 1,
        },
    }

    args.is_pretrain = True
    domain = args.dataset_domain 
    assert domain in ["animal", "phantom", "sim", "real"], f'The domain of dataset ({domain}) must be one of the following:["animal", "phantom", "sim", "real"] '
    if domain == "animal" or domain == "phantom":    
        args.num_classes = 3 
    else: 
        args.num_classes=2
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    
    ckpt_dir = "checkpoints/pt_ani/TU_R50-ViT-B_16_skip0_epo100_bs2_lr0.01_256_s40000/epoch_99.pth" #animal
    net.load_state_dict(torch.load(ckpt_dir))

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    if args.is_savenii:
        args.test_save_dir = 'predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


