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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import MedDataset
from utils.dice_score import dice_loss

seed = 0


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        domain = None
):
    
    dir_checkpoint = Path(f'./{domain}_lr-{learning_rate}_seed{seed}/')
    pt_dir_img = Path(f'../data/{domain}_train/images/')
    pt_dir_mask = Path(f'../data/{domain}_train/masks/')

    pt_test_dir_img = Path(f'../data/{domain}_test/images/')
    pt_test_dir_mask = Path(f'../data/{domain}_test/masks/')


    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 1. Create dataset


    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    train_set = MedDataset(images_dir= pt_dir_img, mask_dir=pt_dir_mask,  domain=domain)
    val_set = MedDataset(images_dir=pt_test_dir_img, mask_dir=pt_test_dir_mask,domain=domain)
    n_train = len(train_set)
    n_val = len(val_set)


    # 2. Create data loaders
    train_loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_loader = DataLoader(train_set, shuffle=True, **train_loader_args)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)
    # (Initialize logging)
    experiment = wandb.init(project=f'U-Net-{domain}', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
         save_checkpoint=save_checkpoint,  amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)#, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_val = -1
    # 4. Begin training
    for epoch in range( epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score, jac, miou = evaluate(model, val_loader, device, amp)
            print(f"Dice:  {val_score}, Jaccard: {jac}, mIoU: {miou}")
            if miou>best_val:

                best_val = miou
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                #state_dict['mask_values'] = val_set.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'best_val.pth'))
                logging.info(f'Best validate Checkpoint saved!')
            scheduler.step(miou)
            #main()
            logging.info('Validation Dice score: {}'.format(val_score))
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'jaccard': jac, 
                    "miou": miou,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass

        if save_checkpoint and epoch%10 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            #state_dict['mask_values'] = val_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
    print("best_val: {}".format(best_val))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', action='store_true',  
                        help='Load model from a .pth file', default = False)
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--dataset_domain', type=str, default="animal", help='Choose one of the following: "phantom", "animal", "sim", "real"')
    parser.add_argument('--ckpt_dir', type=str, 
                        default="", help = "Checkpoints directory to load")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args() 

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # check domain
    domain = args.dataset_domain 
    assert domain in ["animal", "phantom", "sim", "real"], f'The domain of dataset ({domain}) must be one of the following:["animal", "phantom", "sim", "real"] '
    # different domains raise different input and output
    if domain == "animal" or domain == "phantom": 
        n_channels = 3
        args.classes = 3 
    elif domain == 'sim': 
        n_channels = 3
        args.classes=2
    else: 
        n_channels = 1 
        args.classes = 2


    model = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    if args.load:
        ckpt_dir = args.ckpt_dir
        state_dict = torch.load(ckpt_dir, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {ckpt_dir}')


    model.to(device=device)
    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        amp=args.amp, 
        domain = domain
    )

