# Endovascular Medical Challenge

This repository is the implementation of baselines (including UNet, TransUNet and SwinUNet) on the Endovascular Dataset

## Prerequisites

Please install dependence package by run following command:
```
pip install -r requirements.txt
```

## Datasets

We use the Endovascular Dataset obtained from hospitals and laboratories in the UK. The dataset contains X-Ray images come from four modalities, including animal, phantom, simulation and real human. The images are paired with the ground truth annotations. The dataset can be downloaded [here](https://vision.aioz.io/f/7b986782043d403bb50e/). After downloading, please put it into the root folder.

## Checkpoints 

We provide the checkpoints of the baselines, which can be downloaded with [this link](https://vision.aioz.io/f/d6140625c1d24919b753/). Each baseline contains the checkpoint of each modality in the dataset. After downloading, please put it into the root folder.
## Baselines 

### UNet

Navigate to the Unet folder first by running `cd Unet`

To training on the Endovascular data, run the following command: 

```
python train.py --amp --dataset_domain DOMAIN
```

To test on the Endovascular data, run the following command: 

```
python test.py --amp --dataset_domain DOMAIN --ckpt_dir CKPT_DIR
```

Please change the checkpoint directory to load (`CKPT_DIR`) and replace `DOMAIN` (the value of the `--dataset_domain` flag) with one of the following: `phantom`, `animal`, `sim` (simulation), `real`. 


### TransUNet

Navigate to the TransUnet folder first by running `cd TransUnet`

To training on the Endovascular data, run the following command: 

```
python train.py --dataset_domain DOMAIN
```

To test on the Endovascular data, run the following command: 

```
python test.py --dataset_domain DOMAIN --ckpt_dir CKPT_DIR
```

Please change the checkpoint directory to load (`CKPT_DIR`) and replace `DOMAIN` (the value of the `--dataset_domain` flag) with one of the following: `phantom`, `animal`, `sim` (simulation), `real`.

Currenly, we only benchmark TransUNet on this dataset with the `R50-ViT-B-16` backbone architecture. You can try more backbones by modifying the `--arch` flag. 

### SwinUNet

Navigate to the SwinUnet folder first by running `cd SwinUnet`

The pre-trained swin transformer model (Swin-T) can be download [here](https://vision.aioz.io/f/bc969973bbfe49d9bb01/). After downloading, please put it into the `pretrained_ckpt` folder.

To training on the Endovascular data, run the following command: 

```
python train.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --dataset_domain DOMAIN
``` 

To test on the Endovascular data, run the following command: 

```
python test.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --dataset_domain DOMAIN --ckpt_dir CKPT_DIR
```

Please change the checkpoint directory to load (`CKPT_DIR`) and replace `DOMAIN` (the value of the `--dataset_domain` flag) with one of the following: `phantom`, `animal`, `sim` (simulation), `real`.

## Acknowledgment 

The implementation of this repo is based on 
[milesial](https://github.com/milesial/Pytorch-UNet)'s work (UNet), [Chen](https://github.com/Beckschen/TransUNet)'s work (TransUNet) and [Hu Cao](https://github.com/HuCaoFighting/Swin-Unet)'s work (SwinUNet). We thank the authors for sharing the code.

### License

MIT License

