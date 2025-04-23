import torch
import torch.nn as nn
import torchvision.models as models 

class WAAMViD_Netv1(nn.Module):
    def __init__(self):
        super(WAAMViD_Netv1, self).__init__()
        
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.feature_extraction = self.alexnet.features
        self.fc1 = nn.Linear(2330, 4660)
        self.fc2 = nn.Linear(4660, 4660)
        self.fc3 = nn.Linear(4660, 4660)
        self.fc4 = nn.Linear(4660, 4660)
        self.last = nn.Linear(4660, 1)
        self.relu = nn.ReLU()
    
    def forward(self, image, metadata):
        
        features = self.feature_extraction(image)
        
        flat_features = torch.flatten(features, start_dim=1)
        flat_metadata = torch.flatten(metadata, start_dim=1)
        
        x = torch.cat((flat_features, flat_metadata), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        width = self.last(x)

        return width

class WAAMViD_Netv2(nn.Module):
    def __init__(self):
        super(WAAMViD_Netv2, self).__init__()
        
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.feature_extraction = self.alexnet.features
        self.fc1 = nn.Linear(2330, 4660)
        self.fc2 = nn.Linear(4660, 2330)
        self.fc3 = nn.Linear(2330, 1165)
        self.fc4 = nn.Linear(1165, 582)
        self.last = nn.Linear(582, 1)
        self.relu = nn.ReLU()
    
    def forward(self, image, metadata):
        
        features = self.feature_extraction(image)
        
        flat_features = torch.flatten(features, start_dim=1)
        flat_metadata = torch.flatten(metadata, start_dim=1)
        
        x = torch.cat((flat_features, flat_metadata), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        width = self.last(x)

        return width
      
class WAAMViD_Netv3(nn.Module):
    def __init__(self):
        super(WAAMViD_Netv3, self).__init__()
        
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.feature_extraction = self.alexnet.features
        self.fc1 = nn.Linear(2330, 4660)
        self.fc2 = nn.Linear(4660, 4660)
        self.fc3 = nn.Linear(4660, 4660)
        self.last = nn.Linear(4660, 1)
        self.relu = nn.ReLU()
    
    def forward(self, image, metadata):
        
        features = self.feature_extraction(image)
        
        flat_features = torch.flatten(features, start_dim=1)
        flat_metadata = torch.flatten(metadata, start_dim=1)
        
        x = torch.cat((flat_features, flat_metadata), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        width = self.last(x)

        return width

class WAAMViD_Netv4(nn.Module):
    def __init__(self):
        super(WAAMViD_Netv4, self).__init__()
        
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.feature_extraction = self.alexnet.features
        self.fc1 = nn.Linear(2330, 4660)
        self.fc2 = nn.Linear(4660, 9320)
        self.fc3 = nn.Linear(9320, 9320)
        self.fc4 = nn.Linear(9320, 4660)
        self.last = nn.Linear(4660, 1)
        self.relu = nn.ReLU()
    
    def forward(self, image, metadata):
        
        features = self.feature_extraction(image)
        
        flat_features = torch.flatten(features, start_dim=1)
        flat_metadata = torch.flatten(metadata, start_dim=1)
        
        x = torch.cat((flat_features, flat_metadata), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        width = self.last(x)

        return width

class WAAMViD_Netv5(nn.Module):
    def __init__(self):
        super(WAAMViD_Netv5, self).__init__()
        
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.feature_extraction = self.alexnet.features
        self.fc1 = nn.Linear(2330, 4660)
        self.fc2 = nn.Linear(4660, 4660)
        self.fc3 = nn.Linear(4660, 4660)
        self.fc4 = nn.Linear(4660, 4660)
        self.fc5 = nn.Linear(4660, 4660)
        self.last = nn.Linear(4660, 1)
        self.relu = nn.ReLU()
    
    def forward(self, image, metadata):
        
        features = self.feature_extraction(image)
        
        flat_features = torch.flatten(features, start_dim=1)
        flat_metadata = torch.flatten(metadata, start_dim=1)
        
        x = torch.cat((flat_features, flat_metadata), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        width = self.last(x)

        return width
