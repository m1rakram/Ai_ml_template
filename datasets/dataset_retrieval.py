import torch 
import torch.nn
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset




class custom_dataset(Dataset):

    # initialize your dataset class
    def __init__(self, mode ="train", image_path = "relative/path/to/your/images", label_path = "relative/path/to/your/labels"):
        self.mode = mode    # you have to specify which set do you use, train, val or test
        self.image_path = image_path #you may need this var in getitem
        self.label_path = label_path


        #create list of paths for the images: (depends on your dataset structure)

        self.total_images = []
        self.labels = []

        #distribute to val train and test

        val_images = []
        test_images  = []
        train_images = []


        if(mode == "train"):
            self.image_list = train_images
        elif(mode == "val"):
            self.image_list = val_images
        else:
            self.image_list = test_images


    def __getitem__(self, index):
        # getitem is required field for pytorch dataloader. Check the documentation

        image  = Image.open(self.image_list[index])

        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

        label = torch.as_tensor(label)

        # all labels should be converted from any data type to tensor
        # for parallel processing


        return image, label
    


    
    def __len__(self):
        return len(self.image_list)