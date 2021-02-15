import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class KAISTDataset(Dataset):

    def __init__(self,path_images,path_objects,split):
    
      self.split=split
      self.images=[]
      self.objects=[]
      
      f=open("/content/drive/MyDrive/MyCode/train-all-20.txt",'r')
      path=f.readlines()

      # Load images files
      for i in range(len(path)):
        self.images.append(path_images+path[i][:6]+path[i][:11]+"/lwir/"+path[i][11:17]+".jpg")

      # Load annotation files
      with open(path_objects+'TRAIN_objects.json', 'r') as j:
        self.objects = json.load(j)

      assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB') 

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]

        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['is_crowd'])  # (n_objects)
    
        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties,i

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        index=list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
            index.append(b[4])
            
        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties,index