from functools import partial
import argparse
import copy
import glob
import json
import logging
import math
import os
import re
import random
import sys
import time
import numpy as np
from typing import List, Optional
from PIL import Image, ImageFile

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms


sys.path.append("../../dinov2-main")
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.utils import ModelWithNormalize, evaluate, extract_features


def extract_dataset_features(args, cache_name, model, input_dataset):
    # print(os.path.join(args.cache_dir, cache_name))
    if os.path.exists(os.path.join(args.cache_dir, cache_name)):
        with open(os.path.join(args.cache_dir, cache_name), 'r') as f:
            feat_dict = json.load(f)

        features = torch.Tensor(feat_dict['features']).cuda()

        print(f"{cache_name} features loaded!")

    else:
        features, _ = extract_features(
            model, input_dataset, args.batch_size, args.num_workers, gather_on_cpu=False)

        feat_dict = dict()
        feat_dict['features'] = features.detach().cpu().tolist()

        with open(os.path.join(args.cache_dir, cache_name), 'w') as f:
            json.dump(feat_dict, f)
        
        print(f"{cache_name} features saved!")
            
    return features

def compute_similarity(obj_feats, roi_feats):
    """
    Compute Cosine similarity between object features and proposal features
    """
    roi_feats = roi_feats.unsqueeze(-2)
    
    # ########## Count Elapsed Time ##########
    start = time.time()

    sim = torch.nn.functional.cosine_similarity(roi_feats, obj_feats, dim=-1)
    
    end = time.time()
    print(f"Elapsed Time on Similarity Computation: {end-start} seconds.")
    # ########################################
    return sim

def compute_euclidean_distance(obj_feats, roi_feats):
    """
    Compute Euclidean between object features and proposal features
    """
    roi_feats = roi_feats.unsqueeze(-2)
    sim = torch.cdist(roi_feats, obj_feats)
    return sim

def stableMatching(preferenceMat):
    # created by shu
    mDict = dict()
    
    engageMatrix = np.zeros_like(preferenceMat)
    for i in range(preferenceMat.shape[0]):
        tmp = preferenceMat[i]
        sortIndices = np.argsort(tmp)[::-1]
        mDict[i] = sortIndices.tolist()

    freeManList = list(range(preferenceMat.shape[0]))

    while freeManList:
        curMan = freeManList.pop(0)
        curWoman = mDict[curMan].pop(0)
        if engageMatrix[:, curWoman].sum() == 0:
            engageMatrix[curMan, curWoman] = 1
        else:
            engagedMan = np.where(engageMatrix[:, curWoman] == 1)[0][0]
            if preferenceMat[engagedMan, curWoman] > preferenceMat[curMan, curWoman]:
                freeManList.append(curMan)
            else:
                engageMatrix[engagedMan, curWoman] = 0
                engageMatrix[curMan, curWoman] = 1
                freeManList.append(engagedMan)
    return engageMatrix

def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = [],
        add_help: bool = True):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]

    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train_path",
        default="../database/train",
        type=str,
        help="Path to train dataset.",
    )
    parser.add_argument(
        "--test_path",
        default="../database/test",
        type=str,
        help="Path to test dataset.",
    )
    parser.add_argument(
        "--imsize",
        default=224,
        type=int,
        help="Image size",
    )
    parser.add_argument(
        "--pretrained_weights",
        default="dinov2_vits14_pretrain.pth",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to save outputs.")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of data loading workers per GPU.")

    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
             "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )

    parser.set_defaults(
        train_dataset="Object",
        test_dataset="Scene",
        batch_size=1,
        num_workers=0,
    )
    return parser


def eval(output_dir, level, weight_idx, object_features, num_object, num_example, scene_features, scene_dataset, split_idx, score_thresh_predefined, score_thresh):

    results = []
    # reshape scene feature matrix
    scene_cnt = [0, *scene_dataset.cfg['length']]
    scene_idx = [sum(scene_cnt[:i + 1]) for i in range(len(scene_cnt))]
    scene_features_list = [scene_features[scene_idx[i]:scene_idx[i + 1]] for i in range(len(scene_dataset.cfg['length']))]

    proposals = scene_dataset.cfg['proposals']
    proposals_list = [proposals[scene_idx[i]:scene_idx[i + 1]] for i in range(len(scene_dataset.cfg['length']))]
    
    # ============ Step 2: similarity ============
    for idx, scene_feature in enumerate(scene_features_list[split_idx[0]:split_idx[-1]]):
        
        # import time
        # start = time.time()
        
        if isinstance(scene_feature, list):
            scene_feature = torch.Tensor(scene_feature)
        if isinstance(object_features, list):
            object_features = torch.Tensor(object_features)
        
            
        # sim_mat = compute_euclidean_distance(object_features, scene_feature)
        sim_mat = compute_similarity(object_features, scene_feature)
        # sim_mat = torch.mm(object_features,scene_features.T)
         
        
        sim_mat = sim_mat.view(len(scene_feature), num_object, num_example)
        # sim_mat = torch.max(sim_mat, dim=0)[0].unsqueeze(0) - sim_mat

        # option 1
        # sims, _ = torch.max(sim_mat, dim=2)  # choose max score over profile examples of each object instance
        
        # option 2
        # sims = torch.mean(sim_mat, dim=2)  # not good as max
        
        # option 3
        sim_mat, _ = torch.sort(sim_mat, dim=2, descending=True)
        sims = torch.mean(sim_mat[..., :5], dim=2)
        sims[sims < score_thresh] = -1

        proposals = proposals_list[split_idx[0]:split_idx[-1]][idx]

        # empirically make possible background with similarity score = -1
        # bk_idx = []
        # for j in range(len(proposals)):
        #     box_area = proposals[int(j)]['bbox'][-2] * proposals[int(j)]['bbox'][-1]
        #     if box_area > (1/100) * proposals[int(j)]['image_width'] * proposals[int(j)]['image_height'] :
        #         sims[j,:] = -1
        #     elif (proposals[int(j)]['area'] / box_area) < 0.5:
        #         sims[j,:] = -1

        ########################################## Stable Matching Strategy ##########################################
        # ------------ ranking and sorting ------------
        # Initialization
        sel_obj_ids = [str(v) for v in list(np.arange(num_object))]  # ids for selected obj
        sel_roi_ids = [str(v) for v in list(np.arange(len(scene_feature)))]  # ids for selected roi
        
        # Padding
        max_len = max(len(sel_roi_ids), len(sel_obj_ids))
        sel_sims_symmetric = torch.ones((max_len, max_len)) * -1
        sel_sims_symmetric[:len(sel_roi_ids), :len(sel_obj_ids)] = sims.clone()
        
        pad_len = abs(len(sel_roi_ids) - len(sel_obj_ids))
        if len(sel_roi_ids) > len(sel_obj_ids):
            pad_obj_ids = [str(i) for i in range(num_object, num_object + pad_len)]
            sel_obj_ids += pad_obj_ids
        elif len(sel_roi_ids) < len(sel_obj_ids):
            pad_roi_ids = [str(i) for i in range(len(sel_roi_ids), len(sel_roi_ids) + pad_len)]
            sel_roi_ids += pad_roi_ids
        
        # ------------ stable matching ------------
        
        # ########## Count Elapsed Time ##########
        start = time.time()    
        
        matchedMat = stableMatching(
            sel_sims_symmetric.detach().data.numpy())  # matchedMat is raw predMat
        predMat_row = np.zeros_like(
            sel_sims_symmetric.detach().data.numpy())  # predMat_row is the result after stable matching
        Matches = dict()
        for i in range(matchedMat.shape[0]):
            tmp = matchedMat[i, :]
            a = tmp.argmax()
            predMat_row[i, a] = tmp[a]
            Matches[sel_roi_ids[i]] = sel_obj_ids[int(a)]
        # print("Done!")
        
        end = time.time()
        print(f"Elapsed Time on Stable Matching: {end-start} seconds.")
        # ########################################
        
        # ------------ thresholding ------------
        if idx < 120:
            score_thresh = score_thresh_predefined[0]
        else:
            score_thresh = score_thresh_predefined[-1]
        preds = Matches.copy()
        for key, value in Matches.items():           
            if sel_sims_symmetric[int(sel_roi_ids.index(key)), int(sel_obj_ids.index(value))] <= score_thresh:
                del preds[key]
                continue
            
            # # empirical setting for visualization
            # box_area = proposals[int(key)]['bbox'][-2] * proposals[int(key)]['bbox'][-1]
            # if (proposals[int(key)]['area'] / box_area) < 0.4:
            #     del preds[key]
            #     continue
    
        
        # ------------ save results ------------
        for k, v in preds.items():
            result = dict()
            result['image_id'] = proposals[int(k)]['image_id'] - split_idx[0]
            result['category_id'] = int(v)
            result['bbox'] = proposals[int(k)]['bbox']
            result['score'] = float(sims[int(k), int(v)])
            result['image_width'] = proposals[int(k)]['image_width']
            result['image_height'] = proposals[int(k)]['image_height']
            result['scale'] = proposals[int(k)]['scale']
            results.append(result)
        
        # print("Done!")
        ##############################################################################################################
    return results