import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset



class custom_dataset(Dataset):
    def __init__(self, mode = "train", root = "datasets/demo_dataset", transforms = None):
        super().__init__()
        self.mode = mode
        self.root = root
        self.transforms = transforms
        
        #select split
        self.folder = os.path.join(self.root, self.mode)
        
        #initialize lists
        self.image_list = []
        self.label_list = []
        
        #save class lists
        self.class_list = os.listdir(self.folder)
        self.class_list.sort()
        
        for class_id in range(len(self.class_list)):
            for image in os.listdir(os.path.join(self.folder, self.class_list[class_id])):
                self.image_list.append(os.path.join(self.folder, self.class_list[class_id], image))
                label = np.zeros(len(self.class_list))
                label[class_id] = 1.0
                self.label_list.append(label)
        
    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.label_list[index]
        
        
        image = Image.open(image_name)
        if(self.transforms):
            image = self.transforms(image)
        
        label = torch.tensor(label)
        
        return image, label
            
    def __len__(self):
        return len(self.image_list)        


