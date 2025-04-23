import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import GetData
import model
import math


class Dimensional_Analysis(object):
    def __init__(self):
        self.device = torch.device('cuda')
        self.eval_model = model.WAAMViD_Netv2()
        self.eval_model.load_state_dict(torch.load('', map_location = self.device)) # change this path
        self.eval_model = self.eval_model.to(self.device)
        self.eval_model.eval()

        self.transform = A.Compose(
        [
            A.Resize(128, 128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        

    def evaluation(self, video, csv_file):
        dataset = GetData(video=video, csv_file=csv_file, transform=self.transform)

        predict_width_list = []

        test_dict = {}

        for idx in range(len(dataset)):
            data = dataset[idx]
            input1 = data["predict_mask"].unsqueeze(0).to(self.device, dtype=torch.float32)
            input1 = input1.repeat(1, 3, 1, 1)
            input2 = torch.cat([
                data["camera_matrix"].unsqueeze(0).to(self.device, dtype=torch.float32),
                data["distortion_coefficients"].unsqueeze(0).to(self.device, dtype=torch.float32),
                data["rotation_matrix"].unsqueeze(0).to(self.device, dtype=torch.float32),
                data["translation_vector"].unsqueeze(0).to(self.device, dtype=torch.float32)
            ], dim=1)

            with torch.no_grad():
                output = self.eval_model(input1, input2)
                predict_width = math.floor(output.cpu().item() * 10) / 10

                try:
                    test_dict[predict_width] += 1
                except KeyError:
                    test_dict.setdefault(predict_width, 0)

                predict_width_list.append(predict_width)

        print(len(test_dict))
        print(test_dict)

        return predict_width_list