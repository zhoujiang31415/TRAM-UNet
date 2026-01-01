# coding=utf-8
import os
import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from tqdm import tqdm
import pandas as pd


from eval import eval_net
from model.model import UNet
from model.utnet import UTNet
from model.tram_unet import TRAM_UNet
from loss import SoftDiceLoss, BCESoftDiceLoss


from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader
from torch.backends import cudnn

from thop import profile


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.0001,
              save_cp=True,
              img_scale=1.0,
              fold_num=0):

    data_root = 'dataset/BUSBRA/five-fold/'
    # data_root = 'dataset/BUSI/five-fold/'
    # data_root = 'dataset/BLUI/five-fold/'

    
    dir_checkpoint = 'checkpoints/'
    fold = ["fold1", "fold2", "fold3", "fold4", "fold5"]

    train_dir_img = os.path.join(data_root, fold[fold_num - 1], "train", "images/")
    train_dir_mask = os.path.join(data_root, fold[fold_num - 1], "train", "labels/")
    test_dir_img = os.path.join(data_root, fold[fold_num - 1], "test", "images/")
    test_dir_mask = os.path.join(data_root, fold[fold_num - 1], "test", "labels/")

    train_dataset = BasicDataset(train_dir_img, train_dir_mask, 'train', img_scale)
    test_dataset = BasicDataset(test_dir_img, test_dir_mask, 'test', img_scale)
    n_train = len(train_dataset)
    n_val = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,
                            drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = BCESoftDiceLoss()

    # Log Setup
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    dataset_name = data_root.split('/')[1] 
    model_name = getattr(net, 'id', 'UnknownModel') 
    log_file_path = os.path.join(log_dir, f"{model_name}_on_{dataset_name}_fold{fold_num}_log.csv")
    log_data = []
    print(f"Logging training data to: {log_file_path}")


    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
            Images scaling:  {img_scale}
            criterion:       {criterion}
        ''')
    best_acc = 0
    best_dice = 0
    best_iou = 0
    best_sensitive = 0
    best_specificity = 0

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        count = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']


                # Check if the image size is 256x256 

                if imgs.shape[2] != 256 or imgs.shape[3] != 256:
                    print(f"\n[WARNING] Found wrong image size: {imgs.shape}!")
                
                # Check if the Mask value is out of bounds
                mask_max = torch.max(true_masks)
                if mask_max > 1:
                    if count % 100 == 0: 
                        print(f"\n Detected Mask max value {mask_max} > 1. Fixing...")
                    
                    true_masks = true_masks / 255.0
                    true_masks[true_masks > 0.5] = 1
                    true_masks[true_masks <= 0.5] = 0
                    true_masks = true_masks.long() 


                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                

                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                
                # Validation Logic
                if count == (n_train // batch_size) or count == (n_train // batch_size - 1):
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        if value.grad is not None:
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    
                    val_score = eval_net(net, val_loader, device, criterion)
                    
                    val_loss = val_score[0] 
                    acc_score = val_score[1]
                    dc_score = val_score[2]
                    iou_score = val_score[3]
                    sensitive_score = val_score[4]
                    specificity_score = val_score[5]
                    
                    scheduler.step(dc_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(dc_score))
                        writer.add_scalar('Loss/test', dc_score, global_step)
                    else:
                        logging.info('Validation Accuracy: {}'.format(acc_score))
                        logging.info('Validation Dice Coeff: {}'.format(dc_score))
                        logging.info('Validation IoU: {}'.format(iou_score))
                        logging.info('Validation Sensitive: {}'.format(sensitive_score))
                        logging.info('Validation Specificity: {}'.format(specificity_score))
                        
                        writer.add_scalar('Val_Loss', val_loss, global_step)
                        writer.add_scalar('Accuracy/test', acc_score, global_step)
                        writer.add_scalar('Dice/test', dc_score, global_step)
                        writer.add_scalar('IoU/test', iou_score, global_step)
                        writer.add_scalar('Sensitive/test', sensitive_score, global_step)
                        writer.add_scalar('Specificity/test', specificity_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                    
                    # <<< CSV LOGGING >>>
                    avg_train_loss = epoch_loss / count if count > 0 else 0
                    
                    epoch_data = {
                        'fold': fold_num,
                        'epoch': epoch + 1,
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'val_acc': acc_score,
                        'val_dice': dc_score,
                        'val_iou': iou_score,
                        'val_sensitive': sensitive_score,
                        'val_specificity': specificity_score
                    }
                    
                    log_data.append(epoch_data)
                    # Real-time Saving
                    pd.DataFrame(log_data).to_csv(log_file_path, index=False)


                count += 1

        if dc_score > best_dice:
            best_dice = dc_score
            best_acc = acc_score
            best_iou = iou_score
            best_sensitive = sensitive_score
            best_specificity = specificity_score
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'fold{fold_num}.pth')
            logging.info(f'best_acc {best_acc}')
            logging.info(f'best_dc {best_dice}')
            logging.info(f'best_iou {best_iou}')
            logging.info(f'best_sensitive {best_sensitive}')
            logging.info(f'best_specificity {best_specificity}')
        else:
            logging.info(f'best_acc {best_acc}')
            logging.info(f'best_dc {best_dice}')
            logging.info(f'best_iou {best_iou}')
            logging.info(f'best_sensitive {best_sensitive}')
            logging.info(f'best_specificity {best_specificity}')

    writer.close()
    logging.info(f"Training complete. Log saved to {log_file_path}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=80,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
  
    return parser.parse_args()


if __name__ == '__main__':
    fold_num = 5            # Change fold number from 1 to 5
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'')
    logging.info(f'Training fold{fold_num}')
    logging.info(f'Using device {device}')
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Selection
    
    # net = UNet(n_channels=3, n_classes=1, bilinear=True).to(DEVICE)
    # net.id = "UNet" 
    
    # net = UTNet(n_channels=3, base_chan=64, n_classes=1).to(DEVICE)
    # net.id = "UNet+Transformer" 
    
    net = TRAM_UNet(n_channels=3, base_chan=64, n_classes=1).to(DEVICE) 
    net.id = "TRAM_UNet" 

    
    logging.info(f'Network:\t{net.id}\n'
               f'\t{net.n_channels} input channels\n'
               f'\t{net.n_classes} output channels (classes)'
                )
    
    if args.load:
      net.load_state_dict(
          torch.load(args.load, map_location=device)
      )
      logging.info(f'Model loaded from {args.load}')
    
    net.to(device=device)

    try:
      train_net(net=net,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                device=device,
                img_scale=args.scale,
                fold_num=fold_num)
    except KeyboardInterrupt:
      torch.save(net.state_dict(), 'INTERRUPTED.pth')
      logging.info('Saved interrupt')
      try:
          sys.exit(0)
      except SystemExit:
          os._exit(0)
