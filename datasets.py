import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

class CustomTripletFeature(Dataset):
    def __init__(self, pos_feat, neg_feat=None, 
                 num_class=100, num_example=24):
        # this dataset is designed to sample pairs on the fly
        # one sampler for positive classes, one sampler for negative classes
        self.pos_feat = pos_feat
        self.pos_labels = torch.arange(0, num_class).reshape(-1, 1).expand(-1, num_example).flatten()
        self.total_num_pos_feat = num_class * num_example
        
        if neg_feat is not None:
            self.neg_feat = neg_feat # actual bg features
            self.neg_labels = num_class
        
        self.prob_sample_bg = 0.25

    def __len__(self):
        if self.neg_feat is not None:
            return len(self.pos_feat) + len(self.neg_feat) * self.prob_sample_bg
        else:
            return len(self.pos_feat)
    
    def __getitem__(self, idx):
        if self.neg_feat is not None:
            if idx < self.total_num_pos_feat:
                return (self.pos_feat[idx][0],
                       [self.pos_labels[idx]] * len(self.pos_feat[idx][0]))
            else:
                return (self.neg_feat[idx-self.total_num_pos_feat][0],
                       [self.neg_labels[idx-self.total_num_pos_feat]] * len(self.neg_feat[idx-self.total_num_pos_feat][0]))
        else:
            return (self.pos_feat[idx][0],
                    [self.pos_labels[idx]] * len(self.pos_feat[idx][0]))

class CustomImageDatasetPos(Dataset):
    def __init__(self, object_dataset, render_dataset=None, 
                 num_class=100, num_profile_example=24, num_render_example=None):
        # one sampler for positive pairs
        self.object_dataset = object_dataset
        self.object_labels = torch.arange(0, num_class).reshape(-1, 1).expand(-1, num_profile_example).flatten()
        # print("object_labels: ", self.object_labels[:50])
        self.total_num_object_imgs = num_profile_example * num_class
        
        print("object_dataset: ", self.object_dataset[0][0].shape, self.object_labels[1].shape)
        
        # print(len(render_dataset))
        if render_dataset is not None:
            self.render_dataset = render_dataset
            self.render_labels = torch.arange(0, num_class).reshape(-1, 1).expand(-1, num_render_example).flatten()
            print("render_dataset: ", self.render_dataset[0][0].shape, self.render_labels[1].shape)
            # print("render_labels: ", self.render_labels[:50])
        else:
            self.render_dataset = None
        
    def __len__(self):
        if self.render_dataset is not None:
            return len(self.object_dataset) + len(self.render_dataset)
        else:
            return len(self.object_dataset)
    
    def __getitem__(self, idx):
        if self.render_dataset is not None:
            if idx < self.total_num_object_imgs:
                return (self.object_dataset[idx][0], 
                        self.object_labels[idx])
            else:
                return (self.render_dataset[idx-self.total_num_object_imgs][0],
                        self.render_labels[idx-self.total_num_object_imgs])
        else:
            return (self.object_dataset[idx][0], 
                    self.object_labels[idx]) 


class CustomTripletDataset(Dataset):
    def __init__(self, object_dataset, num_class=9691, num_profile_example=10, 
                 render_dataset=None, num_render_example=None):
        # one sampler for positive pairs
        self.object_dataset = object_dataset
        self.object_labels = torch.arange(0, num_class).reshape(-1, 1).expand(-1, num_profile_example).flatten()
        # print("object_labels: ", self.object_labels[:50])
        self.total_num_object_imgs = num_profile_example * num_class
        
        # print("object_dataset: ", self.object_dataset[0][0][0].shape, len(self.object_dataset[0][0]))

        # print(len(render_dataset))
        if render_dataset is not None:
            self.render_dataset = render_dataset
            self.render_labels = torch.arange(0, num_class).reshape(-1, 1).expand(-1, num_render_example).flatten()
            # print("render_dataset: ", self.render_dataset[0][0][0].shape, len(self.render_dataset[0][0]))
            # print("render_labels: ", self.render_labels[:50])
        else:
            self.render_dataset = None

    def __len__(self):
        if self.render_dataset is not None:
            return len(self.object_dataset) + len(self.render_dataset)
        else:
            return len(self.object_dataset)
    
    def __getitem__(self, idx):
        if self.render_dataset is not None:
            if idx < self.total_num_object_imgs:
                return (self.object_dataset[idx][0],
                       [self.object_labels[idx]] * len(self.object_dataset[idx][0]))
            else:
                return (self.render_dataset[idx-self.total_num_object_imgs][0],
                       [self.render_labels[idx-self.total_num_object_imgs]] * len(self.render_dataset[idx-self.total_num_object_imgs][0]))
        else:
            return (self.object_dataset[idx][0],
                   [self.object_labels[idx]] * len(self.object_dataset[idx][0]))