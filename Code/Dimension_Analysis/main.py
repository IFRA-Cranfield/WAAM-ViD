import time
import os
import statistics as s
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
from dataloader import GetData
import model


if __name__ == '__main__':
    training_name = ''  # change this
    log_file_name = f'{training_name}.log'

    # seeging
    utils.seeding(28)

    # hyperparameters
    batch_size = 32
    num_epochs = 200
    lr = 1e-3

    # model
    device = torch.device('cuda')
    training_model = model.WAAMViD_Netv2()
    training_model = training_model.to(device)

    # path
    checkpoint_path = f'{training_name}.pth'
    results = ''

    # logging
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # transform
    train_transform = A.Compose(
        [
            A.Resize(128, 128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    # dataset
    video_dir = ''  # change this
    csv_path = ''  # change this
    csv_name = ''  # change this

    generator = torch.Generator().manual_seed(42)  # set the seed for reproducibility

    dataset = GetData(video_dir=video_dir, 
                      csv_path=csv_path, 
                      csv_name=csv_name,
                      transform=train_transform)
    
    train_size = int(0.7 * len(dataset))  # 70% for training
    valid_size = int(0.1 * len(dataset))  # 10% for validation
    eval_size = len(dataset) - train_size - valid_size  # 20% for testing

    train_dataset, valid_dataset, eval_dataset = random_split(dataset, [train_size, valid_size, eval_size], generator=generator)

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

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    logging.info(f"Dataset Size:\nTrain: {len(train_dataset)} - Valid: {len(valid_dataset)} - Test: {len(eval_dataset)}\n")

    # loss and learning rate
    train_loss = []
    val_loss = []
    learning_rate = []
    epoch_time = []

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(training_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           patience=5, threshold=1e-3,
                                                           min_lr=1e-7, verbose=True)

    # training
    best_valid_loss = float('inf')
    early_stopper = utils.EarlyStopper(patience=10)

    logging.info('Starting training')
    for epoch in range(num_epochs):
        start_time = time.time()

        trainLoss = utils.training_loop(training_model, train_loader, optimizer, criterion)
        validLoss = utils.validate_loop(training_model, valid_loader, criterion)

        scheduler.step(validLoss)

        # saving the best checkpoint
        if validLoss < best_valid_loss:
            logging.info(f"Valid loss improved from {best_valid_loss:2.4f} to {validLoss:2.4f}. Saving checkpoint: {checkpoint_path}")

            best_valid_loss = validLoss
            torch.save(training_model.state_dict(), checkpoint_path)

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

    # valuation
    logging.info('Start evaluation')

    eval_model = training_model
    eval_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    eval_model = eval_model.to(device)
    eval_model.eval()

    rmse_list, mae_list, r2_list = utils.testing_loop(eval_model, eval_loader)
    logging.info(f"RMSE: {np.mean(rmse_list):.4f}, MAE: {np.mean(mae_list):.4f}, R2: {np.mean(r2_list):.4f}")

    # save results to a CSV file
    csv_file_path = os.path.join("evaluation_results.csv")
    results = {
        "RMSE": rmse_list,
        "MAE": mae_list,
        "R2": r2_list
    }
    df = pd.DataFrame(results)
    df.to_csv(csv_file_path, index=False)
