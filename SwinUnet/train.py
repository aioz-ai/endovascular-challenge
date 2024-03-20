import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from cal_metric import jaccard, calculate_miou
import torch.optim as optim
from datasets.med_dataset import RandomGenerator, MedDataset

parser = argparse.ArgumentParser()


                  
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8 ,help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=40000, help='random seed')
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
parser.add_argument('--dataset_domain', type=str, 
                        default="animal", help='Choose one of the following: "phantom", "animal", "sim", "real"')
parser.add_argument('--load', action='store_true',  
                        help='Load model from a .pth file', default = False)
parser.add_argument('--ckpt_dir', type=str, 
                        default="", help = "Checkpoints directory to load")

def trainer(device, args, model, snapshot_path):

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    domain = args.dataset_domain
    db_train = MedDataset(images_dir=f'../data/{domain}_train/images', mask_dir=f'../data/{domain}_train/masks', transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]), domain = domain)
    db_test = MedDataset(images_dir=f'../data/{domain}_test/images', mask_dir=f'../data/{domain}_test/masks', transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]), domain = domain)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = -1
    iterator = tqdm(range(max_epoch), ncols=70)


    experiment = wandb.init(project='Swin-UNet', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=max_epoch, batch_size=batch_size, learning_rate=base_lr,
              save_checkpoint=snapshot_path )
    )
    
    for epoch_num in iterator:
        dice_score= 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
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
            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))


        list_pred = [] 
        list_label =[] 
        for test_sample in tqdm(testloader): 
            image, label = test_sample["image"], test_sample["label"]
            dice_per_sample,pred,lab= test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=None,case=None, z_spacing=1, device = device)
            dice_score += dice_per_sample
            list_pred.append(pred)
            list_label.append(lab) 



        list_pred = torch.cat(list_pred, dim = 0)
        list_label = torch.cat(list_label, dim = 0)
        jacc = jaccard(list_pred, list_label)
        miou = calculate_miou(list_pred, list_label)
        performance = dice_score / len(db_test)
        
        print(f"Dice: {performance}, mIoU: {miou}, Jacc: {jacc}")
        # save the best model (best miou)
        if miou>best_performance: 
            best_performance=miou
            save_mode_path = os.path.join(snapshot_path, 'best_dice.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("Save best model with the best dice score = {} ".format(miou))
        logging.info("Performance at epoch iter {}: {}".format(epoch_num, miou))

        experiment.log({
                    'learning rate': base_lr,
                    "miou": miou,
                    'validation Dice': performance,
                    "jaccard": jacc, 
                    
                    'step': iter_num,
                    'epoch': epoch_num,

                })
        if epoch_num%10==0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
    return "Training Finished!"









if __name__ == "__main__":
    args = parser.parse_args()
    config = get_config(args)
    device = 'cuda:0'
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


    snapshot_path = f"checkpoints/{domain}/Swin_"
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) 
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed)# if args.seed!=1234 else snapshot_path
    

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    net.load_from(config)
    if args.load:
        ckpt_dir = args.ckpt_dir
        state_dict = torch.load(ckpt_dir, map_location=device)
        net.load_state_dict(state_dict)
        print(f'Model loaded from {ckpt_dir}')
    trainer(device, args, net, snapshot_path)