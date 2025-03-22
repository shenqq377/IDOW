import copy
import glob
import json
import os
import re
import random
import sys
import numpy as np
from typing import List, Optional
from PIL import Image, ImageFile

import torch
from torchvision import transforms as pth_transforms



class OWIDDataset(torch.utils.data.Dataset):
    '''
    Profile image of each instance in OWID is grouped into multiple subset for Naive metric learning
    '''
    def __init__(self, data_dir, dataset, transform=None, imsize=None, num_samples=2, aug=False, num_class=9691):

        if dataset == 'Object':
            
            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            num_obj = []
            image_dir = []
            mask_dir = []
            count = []
            flag = []
            
            if num_class < 9691:
                sel_class = sorted(random.sample(list(range(len(source_list))), num_class))
                sel_source_list = [source_list[i] for i in sel_class]
            else:
                sel_source_list = sorted(source_list)
            
            for _, source_dir in enumerate(sel_source_list):
                # print(source_dir)
                num_obj.append(source_dir.split('/')[-1].split('.')[0])
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'rgb/*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                
                mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'mask/*'))
                                     if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                if len(image_paths) < 40:
                    print(source_dir, len(image_paths))
                
                image_splits, mask_splits = split_list(image_paths, mask_paths, 
                                                       group_num=len(image_paths) // num_samples,
                                                       retain_left=True)
                
                if aug:
                    flag.extend([1] * len(image_splits.keys()))
                else:
                    flag.extend([0] * len(image_splits.keys()))

                image_dir.extend(list(image_splits.values()))
                mask_dir.extend(list(mask_splits.values()))
                count.append(len(image_splits.keys()))


            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['mask_dir'] = mask_dir
            cfg['obj_name'] = num_obj  # object lists for Object
            cfg['length'] = count
            cfg['num_samples'] = num_samples # num_samples in sub-group

        elif dataset == 'Scene':

            source_list = sorted(glob.glob(os.path.join(data_dir, '*/rgb/*')))

            num_scene = []
            image_dir = []
            proposals = []
            count = []

            # proposals_on_scales_1_square_mask.json
            with open(os.path.join(os.path.dirname(data_dir),
                                   'proposals_on_' + data_dir.split('/')[-1] + '.json')) as f:
                proposal_json = json.load(f)

            for idx, source_dir in enumerate(source_list):
                scene_name = source_dir.split(data_dir)[-1]
                num_scene.append(scene_name)

                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                count.append(len(image_paths))
                proposals.extend(proposal_json[scene_name])


            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['proposals'] = proposals
            cfg['scene_name'] = num_scene  # scene list for Scene
            cfg['length'] = count

        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        self.cfg = cfg
        self.samples = cfg['image_dir']
        self.transform = transform
        self.imsize = imsize
        self.flag = flag
        self.aug = aug
        # self.means, self.stdevs = self.__get_default_norm__()

    def __len__(self):
        return len(self.samples)
    
    def __get_default_norm__(self):
        
        img_list = []
        for item in self.cfg['image_dir']:
            img = Image.open(item[0])
            img = img.resize((self.imsize, self.imsize), Image.Resampling.LANCZOS)
            img = np.asarray(img)
            img = img[:,:,:,np.newaxis]
            img_list.append(img)
        
        imgs = np.concatenate(img_list, axis=3)
        imgs = imgs.astype(np.float32) / 255.
        means, stdevs = [], []
        for i in range(3):
            pixels = imgs[:,:,i,:].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))
        
        return means, stdevs

    def __getitem__(self, index):
        
        path = self.samples[index]
        if self.cfg['dataset'] == "Scene":
            path = [path]

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        imgs = []
        for p in path:
            with open(p, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                w, h = img.size
            
            # 1 for using RGB-augmentation, otherwise 0 for keeping raw image
            if self.transform is not None:
                if self.flag[index] == 0:
                    img = self.transform[0](img)
                else:
                    img = self.transform[-1](img)
            else:
                # resize to self.imsize
                img = img.resize((self.imsize, self.imsize), Image.Resampling.LANCZOS)
            
            imgs.append(img)
            
        return imgs, index

def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list

def split_list(alist, blist=None, group_num=4, shuffle=True, retain_left=False):
    '''
        split "alist" and "blist" into "group_num" groups
        shuffle: random split, True as default
        retain_left: after splitting into "group_num" groups, 
                     the rest elements put into a separate group
    '''
    if blist is not None and (len(alist) != len(blist)):
        print("len(alist) should be equal to len(blist).")
        sys.exit()

    index = list(range(len(alist)))

    if shuffle: 
        random.shuffle(index) 
    
    elem_num = len(alist) // group_num
    a_lists = {}
    b_lists = {}
    
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        a_lists['set'+str(idx)] = subset(alist, index[start:end])
        if blist is not None:
            b_lists['set'+str(idx)] = subset(blist, index[start:end])
    
    if retain_left and group_num * elem_num != len(index):
        a_lists['set'+str(idx+1)] = subset(alist, index[end:])
        if blist is not None:
            b_lists['set'+str(idx+1)] = subset(blist, index[start:end])
    
    return a_lists, b_lists


class InsDetDataset_v4(torch.utils.data.Dataset):
    """
    For InsDet Dataset [Shen et.al, NeurIPS'23 B&D]
    For No Training

    Args:
        data_dir (str): absolute path to load data
        dataset (str): 'Ojcect' or 'Scene' for different part of OWID dataset
        transform: transform to augment input images
        imsize: resize image if not transform given
        num_profiles: select num_profiles samples from each instance category, if num_profiles=0 for select all samples
        num_repeat: repeat the selected samples for transform only, if keep_raw=True
        aug: if aug=True, use RGB-augmentation, otherwise, no augmentation 
        num_samples: profile image of each instance in OWID is grouped into multiple subsets with num_samples
    """
    def __init__(self, data_dir, dataset, transform=None, imsize=448, num_profiles=0, aug=False, num_samples=1):

        source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

        if dataset == 'Object':

            num_obj = []
            image_dir = []
            mask_dir = []
            count = []
            flag = []

            for _, source_dir in enumerate(source_list):
                # print(source_dir)
                num_obj.append(source_dir.split('/')[-1].split('.')[0])
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'images', '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                # mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'masks', '*'))
                #                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                
                # Each object has 24 raw profile images.
                if len(image_paths) < 24:
                    print(source_dir, len(image_paths))

                if num_profiles:
                    # # ========== sequentially sample ==========
                    # sel_paths = [image_paths[i] for i in range(num_profiles)]
                                  
                    # ========== randomly sample ==========
                    sel_index = sorted(random.sample(list(range(len(image_paths))), num_profiles))
                    sel_paths = [image_paths[i] for i in sel_index]
                else:
                    sel_paths = copy.copy(image_paths)

                image_splits, _ = split_list(sel_paths, None,
                                             group_num=len(sel_paths) // num_samples,
                                             retain_left=True)

                if aug:
                    image_dir.extend(list(image_splits.values()) * 2)
                    # mask_dir.extend([mask_paths[i] for i in sel_index])
                    count.append(len(image_splits.keys()) * 2)
                    flag.extend([0] * len(image_splits.keys()) + [1] * len(image_splits.keys()))
                else:
                    image_dir.extend(list(image_splits.values()))
                    # mask_dir.extend([mask_paths[i] for i in sel_index])
                    count.append(len(image_splits.keys()))                    
                    flag.extend([0] * len(image_splits.keys()))

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            # cfg['mask_dir'] = mask_dir
            cfg['obj_name'] = num_obj  # object lists for Object
            cfg['length'] = count

        elif dataset == 'Scene':

            num_scene = []
            image_dir = []
            proposals = []
            count = []
            flag = []

            # proposals_on_scales_4_square_mask.json
            with open(os.path.join(os.path.dirname(data_dir),
                                   'proposals_on_' + data_dir.split('/')[-1] + '.json')) as f:
                proposal_json = json.load(f)

            for idx, source_dir in enumerate(source_list):
                scene_name = source_dir.split('/')[-1]
                num_scene.append(scene_name)

                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                count.append(len(image_paths))
                flag.extend([0] * len(image_paths))
                proposals.extend(proposal_json[scene_name])


            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['proposals'] = proposals
            cfg['scene_name'] = num_scene  # scene list for Scene
            cfg['length'] = count
        
        elif dataset == 'Bg':

            image_dir = []
            count = []
            flag = []

            for idx, source_dir in enumerate(source_list):
                # print(source_dir)
                if source_dir.endswith('masks'):
                    continue
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, "*"))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                # print(len(image_paths))
                count.append(len(image_paths))
                flag.extend([0] * len(image_paths))

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['scene_name'] = ['bg']  # scene list for Scene
            cfg['length'] = count

        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        self.cfg = cfg
        self.samples = cfg['image_dir']
        self.transform = transform
        self.imsize = imsize
        self.flag = flag
        self.aug = aug
        # self.means, self.stdevs = self.__get_default_norm__()

    def __len__(self):
        return len(self.samples)
    
    # def __get_default_norm__(self):
        
    #     img_list = []
    #     for item in self.cfg['image_dir']:
    #         img = Image.open(item)
    #         img = img.resize((self.imsize, self.imsize), Image.Resampling.LANCZOS)
    #         img = np.asarray(img)
    #         img = img[:,:,:,np.newaxis]
    #         img_list.append(img)
        
    #     imgs = np.concatenate(img_list, axis=3)
    #     imgs = imgs.astype(np.float32) / 255.
    #     means, stdevs = [], []
    #     for i in range(3):
    #         pixels = imgs[:,:,i,:].ravel()
    #         means.append(np.mean(pixels))
    #         stdevs.append(np.std(pixels))
        
    #     return means, stdevs

    def __getitem__(self, index):
        
        path = self.samples[index]
        if self.cfg['dataset'] != "Object":
            path = [path]

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        imgs = []
        for p in path:
            with open(p, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                w, h = img.size
            
            # 1 for using RGB-augmentation, otherwise 0 for keeping raw image
            if self.transform is not None:
                if self.flag[index] == 0:
                    img = self.transform[0](img)
                else:
                    img = self.transform[-1](img)
            else:
                # resize to self.imsize
                img = img.resize((self.imsize, self.imsize), Image.Resampling.LANCZOS)
            
            imgs.append(img)
            
        return imgs, index


class RoboToolsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset, transform=None, imsize=448, num_profiles=0, aug=False, num_samples=1):
        '''
        RoboTools Dataset, can apply online RGB-augmentation on the data.
        Args:
            data_dir (str): absolute path to load data
            dataset (str): 'Ojcect' or 'Scene' for different part of OWID dataset
            transform: transform to augment input images
            imsize: resize image if not transform given
            num_profiles: select num_profiles samples from each instance category, if num_profiles=0 for select all samples
            num_repeat: repeat the selected samples for transform only, if keep_raw=True
            keep_raw: if keep_raw=True, only use num_repeat samples for transform, 
                      if keep_raw=False, use all samples (raw selected samples together with num_repeat samples)
        '''
        if dataset == 'Object':
            
            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            num_obj = []
            image_dir = []
            # mask_dir = []
            count = []
            flag = []

            for _, source_dir in enumerate(source_list):
                print(source_dir)
                num_obj.append(source_dir.split('/')[-1].split('.')[0])
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif)', str(p))])
                
                # mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                #                      if re.search('/*\.(pbm)', str(p))])
                if len(image_paths) < 100:
                    print(source_dir, len(image_paths))
                    
                if num_profiles:
                    # # ========== sequentially sample ==========
                    # sel_paths = image_paths[:num_profiles]
                    # # image_dir.extend(image_paths[:num_profiles])
                    # # mask_dir.extend(mask_paths[:num_profiles])
                                  
                    # ========== randomly sample ==========
                    sel_index = sorted(random.sample(list(range(len(image_paths))), num_profiles))
                    sel_paths = [image_paths[i] for i in sel_index]
                else:
                    sel_paths = copy.copy(image_paths)

                image_splits, _ = split_list(sel_paths, None,
                                             group_num=len(sel_paths) // num_samples,
                                             retain_left=True)

                image_dir.extend(list(image_splits.values()))
                # mask_dir.extend([mask_paths[i] for i in sel_index])
                count.append(len(image_splits.keys()))
                if aug:
                    flag.extend([1] * len(image_splits.keys()))
                else:
                    flag.extend([0] * len(image_splits.keys()))


            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            # cfg['mask_dir'] = mask_dir
            cfg['obj_name'] = num_obj  # object lists for Object
            cfg['length'] = count

        elif dataset == 'Scene':

            source_list = sorted(glob.glob(os.path.join(data_dir, '*/rgb/*')))

            num_scene = []
            image_dir = []
            proposals = []
            count = []
            flag = []

            # proposals_on_scales_1_square_mask.json
            with open(os.path.join(os.path.dirname(data_dir),
                                   'proposals_on_' + data_dir.split('/')[-1] + '.json')) as f:
                proposal_json = json.load(f)

            for idx, source_dir in enumerate(source_list):
                scene_name = '/'.join([t for t in source_dir.split('/') if t not in data_dir.split('/')])
                num_scene.append(scene_name)

                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                count.append(len(image_paths))
                flag.extend([0] * len(image_paths))
                proposals.extend(proposal_json[scene_name])


            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['proposals'] = proposals
            cfg['scene_name'] = num_scene  # scene list for Scene
            cfg['length'] = count

        elif dataset == 'Bg':
            
            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            image_dir = []
            count = []
            flag = []

            for idx, source_dir in enumerate(source_list):
                # print(source_dir)
                if source_dir.endswith('masks'):
                    continue
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, "*"))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                # print(len(image_paths))
                count.append(len(image_paths))
                flag.extend([0] * len(image_paths))

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['scene_name'] = ['bg']  # scene list for Scene
            cfg['length'] = count

        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset))
        
        self.cfg = cfg
        self.samples = cfg['image_dir']
        self.transform = transform
        self.imsize = imsize
        self.flag = flag
        self.aug = aug
        # self.means, self.stdevs = self.__get_default_norm__()

    def __len__(self):
        return len(self.samples)
    
    # def __get_default_norm__(self):
        
    #     img_list = []
    #     for item in self.cfg['image_dir']:
    #         img = Image.open(item)
    #         img = img.resize((self.imsize, self.imsize), Image.Resampling.LANCZOS)
    #         img = img[:,:,:,np.newaxis]
    #         img_list.append(img)
        
    #     imgs = np.concatenate(img_list, axis=3)
    #     imgs = imgs.astype(np.float32) / 255.
    #     means, stdevs = [], []
    #     for i in range(3):
    #         pixels = imgs[:,:,i,:].ravel()
    #         means.append(np.mean(pixels))
    #         stdevs.append(np.std(pixels))
        
    #     return means, stdevs

    def __getitem__(self, index):
        
        path = self.samples[index]
        if self.cfg['dataset'] != "Object":
            path = [path]

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        imgs = []
        for p in path:
            with open(p, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                w, h = img.size
            
            # 1 for using RGB-augmentation, otherwise 0 for keeping raw image
            if self.transform is not None:
                if self.flag[index] == 0:
                    img = self.transform[0](img)
                else:
                    img = self.transform[-1](img)
            else:
                # resize to self.imsize
                img = img.resize((self.imsize, self.imsize), Image.Resampling.LANCZOS)
            
            imgs.append(img)
            
        return imgs, index