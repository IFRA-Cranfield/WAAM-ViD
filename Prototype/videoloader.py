import os
import cv2
import torch
import numpy as np
import albumentations as A
from scipy import ndimage as ndi
from albumentations.pytorch import ToTensorV2 
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import tempfile
import regex as re


class Videoloader(object):
    def __init__(self):
        # model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=self.weights)
        self.model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))

        self.model.load_state_dict(torch.load('', map_location=self.device))  # change this path
        self.model.to(self.device)
        self.model.eval()

        self.imagenet_mean = self.weights.transforms().mean
        self.imagenet_std = self.weights.transforms().std

        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ToTensorV2() 
        ])


    def post_processing(self, mask):
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


    def apply_mask_overlay(self, frame_bgr, mask, color=[0, 255, 0], alpha=0.4):
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


    def video_processing(self, input_video, predict_width_list):
        # video I/O setup
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Cannot open input video")

        # variables for width, height, fps
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # embedding output path and fourcc directly
        video_name = "temp_video"
        output_path_test = tempfile.gettempdir()
        output_filename = os.path.join(output_path_test, f'output_{video_name}.mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_filename,
                            fourcc,
                            fps,
                            (width, height))
        if not out.isOpened(): 
            raise IOError("Cannot open output video for writing")

        # processing loop
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read() # reading frames as BGR
            if not ret: break

            # preprocessing & inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # applying albumentations transform
            transformed = self.transform(image=frame_rgb)
            input_batch = transformed['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_batch)['out'] # getting segmentation data

            # post-processing & overlay
            prob_map = torch.sigmoid(output).squeeze().cpu()
            mask = (prob_map > 0.5).numpy().astype(np.uint8)
            mask = self.post_processing(mask) # Uncomment for post-processing

            overlayed_frame_bgr = self.apply_mask_overlay(frame_bgr, mask, color=[0, 255, 0], alpha=0.4)

            # --- Add Predict Width Text ---
            current_width_value = "N/A"  # Default display value

            if frame_idx < len(predict_width_list):
                raw_data = str(predict_width_list[frame_idx])  # Ensure data is a string
                try:
                    # Try extracting a float in the form 'digits.digit'
                    match = re.search(r'(\d+\.\d)', raw_data)
                    current_width_value = f"{float(match.group(1)):.1f}" if match else raw_data
                except:
                    # Fallback if extraction or conversion fails
                    current_width_value = raw_data if raw_data != "None" else "Error"

            # Format the text for display
            text = f"Predicted Width: {current_width_value} mm"

            # draw text box (using centered logic)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness, padding = 1, 2, 10
            color_text, color_rect = (255, 255, 255), (0, 150, 0)
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            total_text_h = text_h + baseline
            h, w = frame_bgr.shape[:2]
            offset_x, offset_y = 50, 70
            rect_br_x = min(w - 1, w - offset_x)
            rect_br_y = min(h - 1, h - offset_y)
            rect_w = text_w + (2 * padding)
            rect_h = total_text_h + (2 * padding)
            rect_tl_x = max(0, rect_br_x - rect_w)
            rect_tl_y = max(0, rect_br_y - rect_h)
            rect_br_x = rect_tl_x + rect_w # recalculate br based on potentially clipped tl
            rect_br_y = rect_tl_y + rect_h
            text_x = rect_tl_x + padding
            text_y = rect_tl_y + padding + text_h + 3
            cv2.rectangle(overlayed_frame_bgr, (rect_tl_x, rect_tl_y), (rect_br_x, rect_br_y), color_rect, -1)
            cv2.putText(overlayed_frame_bgr, text, (text_x, text_y), font, scale, color_text, thickness, cv2.LINE_AA)

            out.write(overlayed_frame_bgr)

            frame_idx += 1

        # cleanup
        cap.release()
        out.release()

        return output_filename