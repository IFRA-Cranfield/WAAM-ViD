import numpy as np
import json
import os
import cv2
import torch
from torch.utils.data import Dataset

class GetData(Dataset):
    def __init__(self, image_dir, json_path, json_name, transform=None):
        # read json file
        json_file = os.path.join(json_path, json_name) # change json file name
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.image_dir = image_dir # change path
        self.transform = transform

        self.images = {img["id"]: img for img in self.data["images"]}
        self.annotations = {ann["image_id"]: ann for ann in self.data["annotations"]}


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        image_info = list(self.images.values())[idx]
        image_path = os.path.join(self.image_dir, image_info["file_name"])

        # read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        # apply image enhancement techniques
        image = self.image_enhance(image)

        # create mask with empty array
        mask = np.zeros((height, width), dtype=np.uint8)
        annotation = self.annotations.get(image_info["id"], None)

        # extract segmentation from annotation
        if annotation:
            segmentation = annotation["segmentation"]
            for polygon in segmentation:
                poly = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly], 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # clone tensor to save memory usage
        mask = mask.clone().detach().unsqueeze(0).float()

        return image, mask


    def image_enhance(self, image):
        """ For ablation study """
        # histogram equalization
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # l, a, b = cv2.split(lab)
        # l = clahe.apply(l)
        # lab = cv2.merge((l, a, b))
        # image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # return image


        # gamma reduction
        gamma = 0.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)
    

        # sharpen image
        # kernel = np.array([[0, -1, 0],
        #             [-1, 5,-1],
        #             [0, -1, 0]])
        
        # return cv2.filter2D(image, -1, kernel)
