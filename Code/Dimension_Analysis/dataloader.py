import os
import cv2
import torch
import numpy as np
import pandas as pd
import ast
from scipy import ndimage as ndi
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch.utils.data import Dataset


class GetData(Dataset):
    def __init__(self, video_dir, csv_path, csv_name, transform=None):
        self.csv_path = csv_path
        self.csv_path = csv_path
        self.csv_name = csv_name
        self.video_dir = video_dir
        self.transform = transform

        self.df = pd.read_csv(os.path.join(csv_path, csv_name))

        self.video_path_list = self.df.iloc[:, 0].tolist()
        self.camera_matrix_list = self.df.iloc[:, 1].apply(ast.literal_eval).tolist()
        self.distortion_coefficients_list = self.df.iloc[:, 2].apply(ast.literal_eval).tolist()
        self.rotation_matrix_list = self.df.iloc[:, 3].apply(ast.literal_eval).tolist()
        self.translation_vector_list = self.df.iloc[:, 4].apply(ast.literal_eval).tolist()

        self.width_list = self.df.iloc[:, 8].apply(ast.literal_eval).tolist()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))

        self.model.load_state_dict(torch.load('', map_location=self.device))  # Change this path
        self.model.to(self.device)
        self.model.eval()

        imagenet_mean = weights.transforms().mean
        imagenet_std = weights.transforms().std

        if not self.transform:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=imagenet_mean, std=imagenet_std),
                ToTensorV2()
            ])

        # flatten all frames from all videos into a single list
        self.frame_data = self._prepare_frame_data()


    def _prepare_frame_data(self):
        """
        Process all videos and flatten frames into a single list.
        Each frame will have its predict_mask and corresponding width.
        """
        frame_data = []
        for video_idx, video_path in enumerate(self.video_path_list):
            video_path = video_path.replace('\\', '/')
            widths = self.width_list[video_idx]  # widths for the current video
            masks = self.video_processing(video_path)  # predict masks for all frames in the video

            # ensure predict_mask and width are aligned
            for frame_idx, (mask, width) in enumerate(zip(masks, widths)):
                frame_data.append({
                    "predict_mask": mask,
                    "camera_matrix": self.camera_matrix_list[video_idx],
                    "distortion_coefficients": self.distortion_coefficients_list[video_idx],
                    "rotation_matrix": self.rotation_matrix_list[video_idx],
                    "translation_vector": self.translation_vector_list[video_idx],
                    "width": width
                })

        return frame_data


    def __len__(self):
        return len(self.frame_data)


    def __getitem__(self, index):
        frame = self.frame_data[index]

        # Process predict_mask
        predict_mask_array = frame["predict_mask"]
        if self.transform:
            if not isinstance(predict_mask_array, np.ndarray):
                predict_mask_array = np.array(predict_mask_array)

            transformed = self.transform(image=predict_mask_array)
            predict_mask_array = transformed["image"]

        # Process other data
        camera_matrix = torch.tensor(frame["camera_matrix"], dtype=torch.float32)
        distortion_coefficients = torch.tensor(frame["distortion_coefficients"], dtype=torch.float32)
        rotation_matrix = torch.tensor(frame["rotation_matrix"], dtype=torch.float32)
        translation_vector = torch.tensor(frame["translation_vector"], dtype=torch.float32)
        width = torch.tensor([frame["width"]], dtype=torch.float32)  # ensure width is a tensor

        return {
            "predict_mask": predict_mask_array,
            "camera_matrix": camera_matrix,
            "distortion_coefficients": distortion_coefficients,
            "rotation_matrix": rotation_matrix,
            "translation_vector": translation_vector,
            "width": width
        }


    def video_processing(self, video_path):
        """
        Process a video and return predict_mask for each frame.
        """
        cap = cv2.VideoCapture(os.path.join(self.video_dir, video_path))
        if not cap.isOpened():
            raise IOError("Cannot open input video")

        mask_array = []
        while True:
            ret, frame_bgr = cap.read()  # read frames as BGR
            if not ret:
                break

            # preprocessing & inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # applying albumentations transform
            transformed = self.transform(image=frame_rgb)
            input_batch = transformed['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_batch)['out']  # Getting segmentation data

            # post-processing
            prob_map = torch.sigmoid(output).squeeze().cpu()
            mask = (prob_map > 0.5).numpy().astype(np.uint8)
            mask = self.post_processing(mask)

            mask_array.append(mask)

        cap.release()
        return mask_array


    def post_processing(self, mask):
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
