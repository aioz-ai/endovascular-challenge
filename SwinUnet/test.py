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
from datasets.med_dataset import RandomGenerator,MedDataset
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from cal_metric import jaccard, calculate_miou
from torchvision import transforms
parser = argparse.ArgumentParser()
# parser.add_argument('--volume_path', type=str,
#                     default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir

parser.add_argument('--dataset_domain', type=str, 
                        default="animal", help='Choose one of the following: "phantom", "animal", "sim", "real"')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--ckpt_dir', type=str, default = '',  help='checkpoint directory to load') # you can change the checkpoint load here

args = parser.parse_args()
config = get_config(args)


def inference(args, model, test_save_path=None):

    db_test = MedDataset(images_dir=f'../data/{args.domain}_test/images', mask_dir=f'../data/{args.domain}_test/masks', transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]), domain = args.domain)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    dice_score = 0
    test_acc = 0
    list_pred = [] 
    list_label=[]
    for test_sample in tqdm(testloader): 
            image, label = test_sample["image"], test_sample["label"]
            acc, dice_per_sample,pred,lab= test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=None,case=None, z_spacing=1, device = "cuda")
            test_acc+=acc
            dice_score += dice_per_sample
            list_pred.append(pred)
            list_label.append(lab) 



    list_pred = torch.cat(list_pred, dim = 0)
    list_label = torch.cat(list_label, dim = 0)
    jacc = jaccard(list_pred, list_label)
    miou = calculate_miou(list_pred, list_label)
    performance = dice_score / len(db_test)
    accuracy = test_acc/len(db_test)
    print(f"Accuracy: {accuracy} -- Jaccard: {jacc} -- mIoU: {miou} -- Dice score: {performance}")


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

    domain = args.domain
    assert domain in ["animal", "phantom", "sim", "real"], f'The domain of dataset ({domain}) must be one of the following:["animal", "phantom", "sim", "real"] '
    if domain == "animal" or domain == "phantom":    
        args.num_classes = 3 
    else: 
        args.num_classes = 2

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_state_dict(torch.load(args.ckpt_dir))

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


