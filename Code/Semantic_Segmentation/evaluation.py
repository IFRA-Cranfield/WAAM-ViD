import os
from scipy import ndimage as ndi

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataloader import GetData
from metrics import dice_metrics, precision_metrics, recall_metrics
import utils
from torchvision import models

import logging


def apply_mask_overlay(frame_bgr, mask, color=[0, 255, 0], alpha=0.4):
    mask_binary = (mask > 0).astype(np.uint8)
    h, w = frame_bgr.shape[:2]

    if mask_binary.shape != (h, w):
        resized_mask = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = mask_binary

    color_mask_overlay = np.zeros_like(frame_bgr)
    color_mask_overlay[resized_mask == 1] = color
    overlayed_frame = cv2.addWeighted(frame_bgr, 1, color_mask_overlay, alpha, 0)

    return overlayed_frame


def denormalize(image, mean, std):
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = image * std + mean
    image = np.clip(image, 0, 1)

    return (image * 255).astype(np.uint8)


def post_processing(mask):
    """
    The following function is an adaptation made by Kim et al. 2024
    Kim, K.W., Duman, A. and Spezi, E., 2024, November. RGU-Net:
    Computationally Efficient U-Net for Automated Brain Extraction of
    mpMRI with Presence of Glioblastoma. In Research Conference 2024
    (p. 28).

    Original code: https://github.com/KWKIM128/Brain_Extraction/blob/main/code/post_processing.py
    """
    # Threshold to get rid of noise
    mask = np.where(mask >= 0.5, 1, 0)

    labels, n = ndi.measurements.label(mask)
    hist = np.histogram(labels.flat, bins=(n + 1), range=(-0.5, n + 0.5))[0]
    if len(hist) <= 1 or np.all(hist[1:] == 0):
        return mask

    i = np.argmax(hist[1:]) + 1
    mask = (labels != i).astype(np.uint8)

    labels, n = ndi.measurements.label(mask)
    hist = np.histogram(labels.flat, bins=(n + 1), range=(-0.5, n + 0.5))[0]
    if len(hist) <= 1 or np.all(hist[1:] == 0):
        return mask

    i = np.argmax(hist[1:]) + 1
    return (labels != i).astype(np.uint8)


""" Testing loop """
def testing_loop(model, loader, path=None, device=torch.device('cuda')):
    dices = []
    precisions = []
    recalls = []

    os.makedirs(os.path.join(path, "raw_frames"), exist_ok=True)
    os.makedirs(os.path.join(path, "overlay_pm"), exist_ok=True)
    os.makedirs(os.path.join(path, "entropy_map"), exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            image = x.to(device)
            mask = y.to(device)

            # output original image
            raw_path = os.path.join(path, "raw_frames", f"frame_{idx:04d}.jpg")
            x_np = x[0].permute(1, 2, 0).detach().cpu().numpy()

            x_np = denormalize(x_np, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            x_np = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)

            cv2.imwrite(raw_path, x_np) # saving the original image

            # run inferencing for prediction
            test_outputs = model(image)
            test_outputs = torch.sigmoid(test_outputs['out'])

            pm = test_outputs[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]
            pm = pm > 0.5
            pm = post_processing(pm)

            gt = mask[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]

            # overlay predict mask
            overlayed_frame_bgr = apply_mask_overlay(x_np, pm, color=[0, 255, 0], alpha=0.4)

            overlay_path = os.path.join(path, "overlay_pm", f"overlay_{idx:04d}.jpg")
            cv2.imwrite(overlay_path, overlayed_frame_bgr)

            # entropy map
            p = np.clip(pm, 1e-6, 1 - 1e-6)
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            entropy_path = os.path.join(path, "entropy_map", f"entropy_{idx:04d}.npy")
            np.save(entropy_path, entropy)

            # calculating metrics, using roi mask and gt.
            dice = dice_metrics(pm, gt)
            precision = precision_metrics(pm, gt)
            recall = recall_metrics(pm, gt)

            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)

    return dices, precisions, recalls


if __name__ == '__main__':
    utils.seeding(28)

    checkpoint_path = '' # change this path
    out_path = '' # change this path

    log_file_name = '' + '.log' # change this name
    logging.basicConfig(filename = log_file_name, level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

    # define or import your model
    device = torch.device('cuda')
    model = models.segmentation.deeplabv3_resnet50()
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if "aux_classifier" not in k}
    model.load_state_dict(filtered_checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    # transforms
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # dataset
    image_dir = '' # change this path
    json_path = '' # change this path
    json_name = '' # change this name

    dataset = GetData(image_dir = image_dir, 
                      json_path = json_path, 
                      json_name = json_name, 
                      transform=transform)

    eval_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    dices, precisions, recalls = testing_loop(model, eval_loader, out_path)
    logging.info(f'Dice: {np.mean(dices):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}')

    # Save results to a CSV file
    csv_file_name = '' + '.csv' # change this name
    csv_file_path = os.path.join(out_path, csv_file_name)
    results = {
        "Dice (%)": [dice * 100 for dice in dices],
        "Precision (%)": [precision * 100 for precision in precisions],
        "Recall (%)": [recall * 100 for recall in recalls]
    }
    df = pd.DataFrame(results)
    df.to_csv(csv_file_path, index=False)
