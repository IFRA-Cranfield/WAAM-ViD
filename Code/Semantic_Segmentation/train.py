import time
import os
import statistics as s
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.data import random_split, Dataset
import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2

from glob import glob

import utils
from dataloader import GetData
from metrics import DiceLoss, DiceBCELoss
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

if __name__ == '__main__':
    training_name = 'test' # change this
    log_file_name = f'{training_name}.log'
    # seeding
    utils.seeding(28)
    
    # hyperparameters 
    batch_size = 32
    num_epochs = 200
    lr = 1e-4
    
    # model
    device = torch.device('cuda')
    model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
    model = model.to(device)
    
    # load previous trained model
    previous_checkpoint_path = '' # change this path
    previous_checkpoint_name = '' # change this name
    previous_checkpoint = os.path.join(previous_checkpoint_path, previous_checkpoint_name)

    previous_checkpoint_model = torch.load(previous_checkpoint, map_location=device)
    model.load_state_dict(previous_checkpoint_model)

    # path
    checkpoint_path = f'{training_name}.pth'
    results = ''
    
    # logging
    logging.basicConfig(filename = log_file_name, level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

    # transform
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
    
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
    
    # dataset
    image_dir = '' # change this path
    json_path = '' # change this path
    json_name = '' # change this name

    generator = torch.Generator().manual_seed(42)  # set the seed for reproducibility

    dataset = GetData(image_dir = image_dir, 
                      json_path = json_path, 
                      json_name = json_name, 
                      transform=train_transform)
    train_size = int(0.8 * len(dataset)) # 80% for training
    valid_size = len(dataset) - train_size # 20% for validation

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=generator)

    # dataset loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
        )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
        )

    logging.info(f'Dataset Size:\nTrain: {len(train_dataset)} - Valid: {len(valid_dataset)}\n')
    
    # loss and learning rate
    train_loss = []
    val_loss = []
    learning_rate = []
    epoch_time = []
    
    summary(model, input_size=(1, 3, 256, 256))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           patience=5, threshold=1e-3,
                                                           min_lr=1e-7, verbose=True)
    loss_fn = DiceBCELoss(weight_dice=0.5, weight_bce=0.5)
    
    # training
    best_valid_loss = float('inf')
    early_stopper = utils.EarlyStopper(patience=10)
    
    logging.info('Starting training')
    for epoch in range(num_epochs):
        start_time = time.time()
        
        trainLoss = utils.training_loop(model, train_loader, optimizer, loss_fn)
        validLoss = utils.validate_loop(model, valid_loader, loss_fn)
        
        scheduler.step(validLoss) 
        
        # saving the best checkpoint
        if validLoss < best_valid_loss:
            logging.info(f"Valid loss improved from {best_valid_loss:2.4f} to {validLoss:2.4f}. Saving checkpoint: {checkpoint_path}")

            best_valid_loss = validLoss
            torch.save(model.state_dict(), checkpoint_path)
            
        # calculate epoch time
        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        epoch_time.append(end_time - start_time)

        logging.info(f"""
                    Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n
                    \tTrain Loss: {trainLoss:.3f}\n
                    \t Val. Loss: {validLoss:.3f}\n
                    """)
        
        # for graph
        train_loss.append(trainLoss)
        val_loss.append(validLoss)
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rate.append(current_lr)
        
        # early stopping
        if early_stopper.early_stop(validLoss):             
            break
        
    # saving the epoch time usage
    for idx, each_epoch_time in enumerate(epoch_time):
        mins = int(each_epoch_time / 60)
        secs = int(each_epoch_time - (mins * 60))
        logging.info(f'Epoch Time {idx + 1}: {mins}m {secs}s\n')

    mean_epochtime = s.mean(epoch_time)
    mean_mins = int(mean_epochtime / 60)
    mean_secs = int(mean_epochtime - (mean_mins * 60))
    logging.info(f'Average Epoch Time: {mean_mins}m {mean_secs}s\n')

    # plot loss and learning rate
    utils.plot_loss(train_loss, val_loss)
    utils.plot_lr(learning_rate)
