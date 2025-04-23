import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from scipy import ndimage as ndi
from albumentations.pytorch import ToTensorV2 
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch.special


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


def extract_segmentation(frame_rgb, mask, frame_idx, output_dir):
    mask_binary = (mask > 0).astype(np.uint8)
    h, w = frame_rgb.shape[:2]

    if mask_binary.shape != (h, w):
        resized_mask = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = mask_binary  

    segmented_part_rgb = cv2.bitwise_and(frame_rgb, frame_rgb, mask=resized_mask)
    seg_dir = os.path.join(output_dir, "segmented_frames")

    os.makedirs(seg_dir, exist_ok=True)
    seg_filename = os.path.join(seg_dir, f"segmented_frame_{frame_idx:05d}.png")
    segmented_part_bgr = cv2.cvtColor(segmented_part_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(seg_filename, segmented_part_bgr)

    if not success:
         print(f"Warning: Failed to save segmented frame {frame_idx}")


def video_processing(video_dir, video_path, result_dir):
    # video I/O setup
    cap = cv2.VideoCapture(os.path.join(video_dir, video_path))
    if not cap.isOpened(): raise IOError("Cannot open input video")

    # variables for width, height, fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # embedding output path and fourcc directly
    video_name = video_path.split('/')[-1]
    output_filename = f'output_{video_name}'
    out = cv2.VideoWriter(os.path.join(result_dir, output_filename),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps if fps > 0 else 30, # Use default fps if needed
                        (width, height))
    if not out.isOpened():
        raise IOError("Cannot open output video for writing")

    # processing loop
    frame_idx = 0
    entropy_list = []
    while True:
        ret, frame_bgr = cap.read() # reading frames as BGR
        if not ret: break

        # preprocessing & inference
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # applying albumentations transform
        transformed = transform(image=frame_rgb)
        input_batch = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)['out'] # getting segmentation data

        # post-processing & overlay
        prob_map = torch.sigmoid(output).squeeze().cpu()
        mask = (prob_map > 0.5).numpy().astype(np.uint8)
        mask = post_processing(mask)

        # compute entropy
        current_entropy = torch.special.entr(prob_map + 1e-8).mean()
        entropy_list.append(current_entropy)

        overlayed_frame_bgr = apply_mask_overlay(frame_bgr, mask, color=[0, 255, 0], alpha=0.4)
        extract_segmentation(frame_rgb, mask, frame_idx, result_dir)

        out.write(overlayed_frame_bgr)

        frame_idx += 1
        print(f'Frame {frame_idx}/{frame_count}', end='\r')

    # cleanup
    cap.release()
    out.release()
    print(f"\nProcessing complete. Output saved.")
    print(f"video saved at: {os.path.join(result_dir, output_filename)}")

    #print average entropy
    if entropy_list:
        avg_entropy = torch.mean(torch.stack(entropy_list)).item()

        csv_file_path = os.path.join(result_dir, "entropy_results.csv")
        data = {"Video": [video_name], "Average Entropy": [avg_entropy]}
        df = pd.DataFrame(data)

        if not os.path.exists(csv_file_path):
            df.to_csv(csv_file_path, index=False)
        else:
            df.to_csv(csv_file_path, mode='a', header=False, index=False)

    else:
        print("No frames processed.")


def csv_reader():
    csv_path = '' # Change this path
    csv_name = '' # Change this name

    df = pd.read_csv(os.path.join(csv_path, csv_name))

    video_path = df.iloc[:, 0].tolist()

    return video_path


if __name__ == '__main__':
    # model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))

    model.load_state_dict(torch.load("", map_location=device)) # Change this path
    model.to(device)
    model.eval()

    imagenet_mean = weights.transforms().mean
    imagenet_std = weights.transforms().std

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2() 
    ])

    video_dir = '' # Change this path
    result_dir = '' # Change this path

    video_path_list = csv_reader()
    for video_path in video_path_list:
        video_path = video_path.replace('\\', '/')
        video_processing(video_dir, video_path, result_dir)
