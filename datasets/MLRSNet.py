import os
import pickle
import pprint
import random
import warnings

import cv2 as cv
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyjson5 as json
import rasterio
from rasterio import features
import torch
from tqdm import tqdm

import albumentations as A

warnings.simplefilter("ignore")
'''
Data loading for the ForestNet Dataset published in:
Irvin, Jeremy, et al. "Forestnet: Classifying drivers of deforestation in indonesia using deep learning on satellite imagery."
'''

class MLRSNet(torch.utils.data.Dataset):
    def __init__(self, configs, mode = 'train'):
        self.configs = configs
        self.root_path = os.path.join(configs['dataset_root_path'],'MLRSNet',"dataset")
        self.mode = mode
        self.annotation_files = os.listdir(os.path.join(self.root_path,"Labels")) 
        self.num_labels = len(self.annotation_files)
        self.label_keys = None
        self.samples = []
        for annotation_file in self.annotation_files:
            if annotation_file.endswith(".csv"):
                annotation = pd.read_csv(os.path.join(self.root_path,"Labels",annotation_file))
                if self.label_keys is None:
                    self.label_keys = annotation.columns.tolist()[1:]
                for index, row in annotation.iterrows():
                    sample = {}
                    sample['label'] = row.tolist()[1:]
                    sample['path'] = os.path.join(self.root_path,"Images",row['image'])   
                    if self.mode == "val":
                        if "2.jpg" in sample['path']:               
                            self.samples.append(sample)
                    elif self.mode == "test":
                        if "3.jpg" in sample['path']:               
                            self.samples.append(sample)
                    else:
                        if "2.jpg" not in sample['path'] and "3.jpg" not in sample['path']:
                            self.samples.append(sample)
        #shuffle samples
        random.Random(999).shuffle(self.samples)
        self.num_examples = len(self.samples)
    
    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv.imread(sample['path'])
        #if image.shape[0] != 256 or image.shape[1] != 256:
        #    image = cv.resize(image,(256,256))
        #    print(index)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv.resize(image,(256,256))
        image = einops.rearrange(image,'h w c -> c h w')
        image = image/255.0
        image = torch.tensor(image,dtype=torch.float32)
        
        label = torch.from_numpy(np.asarray(sample['label']))
        return image, label
    