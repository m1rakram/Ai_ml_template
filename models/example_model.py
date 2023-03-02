import torch
import torch.nn as nn




class ExModel(nn.Module):
    
    def __init__(self, n_classes):
        
        # as images are in RGB, they have originally 3 channels,
        # all channels should be speicified beforehand 
        self.feature_extractor =  nn.Conv2d(in_channels=3, out_channels=128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.classifier_layer = nn.Linear(in_features = 128*400*400, out_features=n_classes)


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.classifier_layer(x)

        return x