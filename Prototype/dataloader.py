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
    def __init__(self, video, csv_file, transform=None):
        self.csv_file = csv_file
        self.video = video
        self.transform = transform

        self.df = pd.read_csv(self.csv_file)

        self.camera_matrix_list = self.df.iloc[:, 0].apply(ast.literal_eval).tolist()
        self.distortion_coefficients_list = self.df.iloc[:, 1].apply(ast.literal_eval).tolist()
        self.rotation_matrix_list = self.df.iloc[:, 2].apply(ast.literal_eval).tolist()
        self.translation_vector_list = self.df.iloc[:, 3].apply(ast.literal_eval).tolist()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))

        self.model.load_state_dict(torch.load('', map_location=self.device))  # change this path
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

        # Flatten all frames from all videos into a single list
        self.frame_data = self._prepare_frame_data()


    def _prepare_frame_data(self):
        """
        Process all videos and flatten frames into a single list.
        Each frame will have its predict_mask and corresponding width.
        """
        frame_data = []

        masks = self.video_processing()  # predict masks for all frames in the video
        for frame_idx, mask in enumerate(masks): # each frame in the video
            frame_data.append({
                "predict_mask": mask,
                "camera_matrix": self.camera_matrix_list[0],
                "distortion_coefficients": self.distortion_coefficients_list[0],
                "rotation_matrix": self.rotation_matrix_list[0],
                "translation_vector": self.translation_vector_list[0]
            })
        return frame_data


    def __len__(self):
        return len(self.frame_data)


    def __getitem__(self, index):
        frame = self.frame_data[index]

        # process predict_mask
        predict_mask_array = frame["predict_mask"]
        if self.transform:
            if not isinstance(predict_mask_array, np.ndarray):
                predict_mask_array = np.array(predict_mask_array)

            transformed = self.transform(image=predict_mask_array)
            predict_mask_array = transformed["image"]

        # process other data
        camera_matrix = torch.tensor(frame["camera_matrix"], dtype=torch.float32)
        distortion_coefficients = torch.tensor(frame["distortion_coefficients"], dtype=torch.float32)
        rotation_matrix = torch.tensor(frame["rotation_matrix"], dtype=torch.float32)
        translation_vector = torch.tensor(frame["translation_vector"], dtype=torch.float32)

        return {
            "predict_mask": predict_mask_array,
            "camera_matrix": camera_matrix,
            "distortion_coefficients": distortion_coefficients,
            "rotation_matrix": rotation_matrix,
            "translation_vector": translation_vector
        }


    def video_processing(self):
        """
        Process a video and return predict_mask for each frame.
        """
        cap = cv2.VideoCapture(self.video)
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
                output = self.model(input_batch)['out']  # getting segmentation data

            # post-processing
            prob_map = torch.sigmoid(output).squeeze().cpu()
            mask = (prob_map > 0.5).numpy().astype(np.uint8)
            mask = self.post_processing(mask)

            mask_array.append(mask)

        cap.release()
        return mask_array


    def post_processing(self, mask):
        """
        The following function is an adaptation made by Kim et al. 2024
        Kim, K.W., Duman, A. and Spezi, E., 2024, November. RGU-Net:
        Computationally Efficient U-Net for Automated Brain Extraction of
        mpMRI with Presence of Glioblastoma. In Research Conference 2024
        (p. 28).

        Original code: https://github.com/KWKIM128/Brain_Extraction/blob/main/code/post_processing.py
        """
        # threshold to get rid of noise
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