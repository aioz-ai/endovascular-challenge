import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from cal_metric import jaccard, calculate_miou
from evaluate import evaluate
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets.med_dataset import RandomGenerator, MedDataset
def trainer(device, args, model, snapshot_path):
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    domain = args.dataset_domain
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    db_train = MedDataset(images_dir=f'../data/{domain}_train/images', mask_dir=f'../data/{domain}_train/masks', transform=transforms.Compose(
                                    [RandomGenerator(args.seed, output_size=[args.img_size, args.img_size])]),domain=domain)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    db_test = MedDataset(images_dir=f'../data/{domain}_test/images', mask_dir=f'../data/{domain}_test/masks', transform=transforms.Compose(
                                    [RandomGenerator(args.seed, output_size=[args.img_size, args.img_size])]), domain=domain)
    print("The length of train set is: {}".format(len(db_train)))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = -1 # for choosing best model
    iterator = tqdm(range(max_epoch), ncols=70)

    experiment = wandb.init(project=f'TransU-Net-{domain}', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=max_epoch, batch_size=batch_size, learning_rate=base_lr,
              save_checkpoint=snapshot_path )
    )




    for epoch_num in iterator:
        model.train()
        dice_score = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            experiment.log({
                    'train loss': loss.item(),
                    'step': iter_num,
                    'epoch': epoch_num
                })
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            

        list_pred = [] 
        list_label =[] 
        for test_sample in tqdm(testloader): 
            image, label = test_sample["image"], test_sample["label"]
            dice_per_sample,pred,lab= test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                    test_save_path=None,case=None, z_spacing=1, device = device)
            dice_score += dice_per_sample
            list_pred.append(pred)
            list_label.append(lab) 

        list_pred = torch.stack(list_pred, dim=0).squeeze()
        list_label = torch.stack(list_label, dim = 0).squeeze()

        jacc = jaccard(list_pred, list_label)
        miou = calculate_miou(list_pred, list_label)
        dice_score = dice_score / len(db_test)

        if miou>best_performance: 
            best_performance=miou
            save_mode_path = os.path.join(snapshot_path, 'best_miou.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("Save best model with the best miou score = {} ".format(dice_score))
        
        print("Performance at epoch iter {}: {} - {} - {}".format(epoch_num, dice_score, miou, jacc))
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        experiment.log({
                    'learning rate': lr_,
                    "miou": miou,
                    'validation Dice': dice_score,
                    "jaccard": jacc, 
                    
                    'step': iter_num,
                    'epoch': epoch_num,
                })
        save_interval = 20 
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    #writer.close()
    return "Training Finished!"





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--load', action='store_true',  
                        help='Load model from a .pth file', default = False)
    parser.add_argument('--max_epochs', type=int,
                        default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=256, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=40000, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=0, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--dataset_domain', type=str, 
                        default="animal", help='Choose one of the following: "phantom", "animal", "sim", "real"')
    parser.add_argument('--ckpt_dir', type=str, 
                        default="", help='Checkpoint directory to load')
    args = parser.parse_args()
    device = "cuda:0"
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

    domain = args.dataset_domain 
    assert domain in ["animal", "phantom", "sim", "real"], f'The domain of dataset ({domain}) must be one of the following:["animal", "phantom", "sim", "real"] '
    

    if domain == "animal" or domain == "phantom":    
        args.num_classes = 3 
    else: 
        args.num_classes=2
 
    snapshot_path = f"checkpoints/{domain}/TU_"
    snapshot_path += args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) 
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) #if args.seed!=1234 else snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
   
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    if args.load:
        ckpt_dir = args.ckpt_dir
        state_dict = torch.load(ckpt_dir, map_location=device)
        net.load_state_dict(state_dict)
        print(f'Model loaded from {ckpt_dir}')
    trainer(device, args, net, snapshot_path)