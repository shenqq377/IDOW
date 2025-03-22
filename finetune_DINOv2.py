import os
import copy
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.nn import init
from datasets import CustomTripletDataset 
from utils import AllPositivePairSelector, HardNegativePairSelector
from utils import FunctionNegativeTripletSelector, hardest_negative

def init_weights(net, net_name=None, init_type='0-normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == '0-normal':
                init.normal_(m.weight.data, 0., gain)
            elif init_type == '1-normal':
                init.normal_(m.weight.data, 1., gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # init.uniform_(m.weight.data, 1.0, gain)
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network {} with {}'.format(net_name, init_type))
    net.apply(init_func)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    # def loss(self, x, y):
    #     raise NotImplementedError
    
    # def train(self, x, y):
    #     raise NotImplementedError
    
    # def eval(self, x, y):
    #     raise NotImplementedError

    
class feature_level_D_ce(BaseModel):
    def __init__(self, input_size=384, middle_size=99, num_class=100):
        super(feature_level_D_ce, self).__init__()
        # input and output size of the feature
        self.model = nn.Sequential(
            nn.Linear(input_size, middle_size),
            # nn.GELU(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(middle_size, num_class),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class TrainerWithCE(nn.Module):
    """
    Add one Linear Layer for K-way or (K+1)-way classification
    """
    def __init__(self, args, model, object_dataset, proposal_dataset, bg_dataset=None, render_dataset=None, num_workers=4, input_size=384) -> None:
        super(TrainerWithCE, self).__init__()
        # args is dictionary type here
        self.args = args
        self.model = model.to(args.device)
        self.model.train()
        self.freeze_layers()
        self.num_class = len(object_dataset.cfg['obj_name']) # num_class for objects (or positive samples)
        if bg_dataset is not None:
            classifier_dim = self.num_class + 1
        else:
            classifier_dim = self.num_class
        self.classifier = feature_level_D_ce(input_size=input_size, middle_size=256, 
                                             num_class=classifier_dim).to(args.device)
        
        
        num_profile_example = len(object_dataset.samples) // len(object_dataset.cfg['obj_name'])
        if render_dataset is not None:
            num_render_example = len(render_dataset.samples) // len(object_dataset.cfg['obj_name'])
        else:
            num_render_example = None

        self.training_data = CustomTripletDataset(
            object_dataset, num_class=len(object_dataset.cfg['obj_name']), num_profile_example=num_profile_example,
            render_dataset=render_dataset, num_render_example=num_render_example
            )

        self.train_dataloader = DataLoader(
            self.training_data, batch_size=self.args.batch_size, shuffle=True, drop_last=True, 
            num_workers=num_workers, pin_memory=True,
            )

        #print("number of training examples: ", len(self.training_data) * args.num_samples)

        # ==================== evaluation data ====================
        
        self.eval_object_dataloader = DataLoader(
            object_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        self.eval_proposal_dataloader = DataLoader(
            proposal_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        if render_dataset is not None:
            self.eval_render_dataloader = DataLoader(
                render_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                num_workers=num_workers, pin_memory=True,
                )
        else:
            self.eval_render_dataloader = None

        
        # 0.1 so far works the best
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.005)
        
        # total number of bg based
        # self.negative_portion = 0.01
        # self.negative_sample = int(bg_feat.shape[0] * self.negative_portion)
        
        # batch size based
        if bg_dataset is not None:
            self.negative_portion = 0.25
            self.negative_sample = int(args.batch_size * args.num_samples * self.negative_portion)
            print("Number of bg examples: {}".format(self.negative_sample))

            self.bg_dataloader = DataLoader(
                bg_dataset, batch_size=self.negative_sample, shuffle=True, drop_last=True,
                num_workers=num_workers, pin_memory=True,
                )
            self.bg_iter = iter(self.bg_dataloader)
        else:
            self.negative_portion = 0
            self.negative_sample = int(args.batch_size * self.negative_portion)            
            self.bg_dataloader = None
            self.bg_iter = None

        self.bg_dataset = bg_dataset

        self.negative_class_label = len(object_dataset.cfg['obj_name'])

    def sample_negative(self):
        try:
            neg_data = next(self.bg_iter)
        except StopIteration:
            self.bg_iter = iter(self.bg_dataloader)  # Reset the data loader
            neg_data = next(self.bg_iter)
        return neg_data[0]

    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            if not 'blocks.11.mlp.fc2' in name:
                param.requires_grad = False
    
    def train(self):
        for i in range(self.args.num_epochs):
            # print(self.training_data.pos_classes[:50])
            for batch_ndx, sample in enumerate(self.train_dataloader):
                pos, pos_labels = sample
                pos_labels = torch.cat(pos_labels)
                # print(len(pos), (pos[0]-pos[1]).sum(), pos_labels)
                pos = torch.stack(pos).view(-1, *pos[0][0].shape)
                pos = pos.to(self.args.device, non_blocking=True)
                pos_feat = self.model(pos)
                all_embeddings = pos_feat
                all_labels = pos_labels
                
                # sample negative
                if self.negative_sample > 0:
                    neg = self.sample_negative()
                    neg = neg[0].to(self.args.device, non_blocking=True)
                    neg_feat = self.model(neg)
                    neg_labels = torch.tensor(
                            [self.negative_class_label] * self.negative_sample, dtype=pos_labels.dtype,
                            device=pos_labels.device)
                    
                    all_embeddings = torch.cat((pos_feat, neg_feat), dim=0)
                    all_labels = torch.cat((pos_labels, neg_labels), dim=0)
                
                self.optimizer.zero_grad()
                all_predictions = self.classifier(all_embeddings)
                loss = torch.nn.functional.cross_entropy(all_predictions, all_labels.to(self.args.device, non_blocking=True))
                print("epoch: {}, batch: {}, loss: {}".format(i, batch_ndx, loss))
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.empty_cache()
            # self.scheduler.step()
            
            if (i+1) % self.args.save_epochs == 0:
                output_f = "epoch{}.pt".format(i+1)
                self.save_weights(i+1, os.path.join(self.args.weights_root, output_f))

    def save_weights(self, epoch, output_f):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, output_f)

    def load_weights(self, f_p):
        checkpoint = torch.load(f_p)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self):
        self.model.eval()
        object_feat = []
        proposal_feat = []
        with torch.no_grad():
            print("computing object features!!")
            for batch_sample in self.eval_object_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)
                object_feat.append(batch_feat.detach().cpu())

            object_feat = torch.cat(object_feat, dim=0)
            print("object_feat1: ", len(object_feat))
            if self.eval_render_dataloader is not None:
                print("computing rendered features!!")
                render_feat = []
                for batch_sample in self.eval_render_dataloader:
                    batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                    render_feat.append(batch_feat.detach().cpu())
                render_feat = torch.cat(render_feat, dim=0)
                # concatenate render feat witg object feat
                object_feat = object_feat.reshape(self.num_class, len(object_feat)//self.num_class, -1)
                render_feat = render_feat.reshape(self.num_class, len(render_feat)//self.num_class, -1)
                
                object_feat = torch.cat((object_feat, render_feat), dim=1).reshape(-1, render_feat.shape[-1])
            
            print("object_feat: ", len(object_feat))
            
            print("computing proposal features!!")
            for batch_sample in self.eval_proposal_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)
                proposal_feat.append(batch_feat.detach().cpu())
            
            proposal_feat = torch.cat(proposal_feat, dim=0)
            print("proposal_feat: ", len(proposal_feat))
            
            return object_feat, proposal_feat

class TrainerWithContrastive(nn.Module):
    """
    Finetuning model with Constrastive Loss
    """
    def __init__(self, args, model, object_dataset, proposal_dataset, bg_dataset=None, render_dataset=None, num_workers=4) -> None:
        super(TrainerWithContrastive, self).__init__()
        # args is dictionary type here
        self.args = args
        self.model = model.to(args.device)
        self.model.train()
        self.freeze_layers()
        self.num_class = len(object_dataset.cfg['obj_name'])
        

        num_profile_example = len(object_dataset.samples) // self.num_class
        if render_dataset is not None:
            num_render_example = len(render_dataset.samples) // self.num_class
        else:
            num_render_example = None

        self.training_data = CustomTripletDataset(
            object_dataset, num_class=self.num_class, num_profile_example=num_profile_example,
            render_dataset=render_dataset, num_render_example=num_render_example
            )

        self.train_dataloader = DataLoader(
            self.training_data, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True,
            )
        
        #print("number of training examples: ", len(self.training_data) * args.num_samples)
        
        # ================================================================================
        self.eval_object_dataloader = DataLoader(
            object_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        self.eval_proposal_dataloader = DataLoader(
            proposal_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        if render_dataset is not None:
            self.eval_render_dataloader = DataLoader(
                render_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                num_workers=num_workers, pin_memory=True,
                )
        else:
            self.eval_render_dataloader = None
        
        
        # 0.1 so far works the best
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.005)
        
        # total number of bg based
        # self.negative_portion = 0.01
        # self.negative_sample = int(bg_feat.shape[0] * self.negative_portion)
        
        # batch size based
        if bg_dataset is not None:
            self.negative_portion = 0.25
            self.negative_sample = int(args.batch_size * args.num_samples * self.negative_portion)
            print("Number of bg examples: {}".format(self.negative_sample))
        
            self.bg_dataloader = DataLoader(
                bg_dataset, batch_size=self.negative_sample, shuffle=True, drop_last=True, 
                num_workers=num_workers, pin_memory=True,
                )
            self.bg_iter = iter(self.bg_dataloader)
            # self.bg_dataset = bg_dataset
            
            self.negative_class_label = self.num_class

        else:
            self.negative_sample = 0
        
        # pair selector
        if bg_dataset is not None:
            self.contrastive_sampler = HardNegativePairSelector()
        else:
            self.contrastive_sampler = AllPositivePairSelector(balance=True)
        
        self.mixup_sampling = False
        self.mixup_ratio = 0.5
        self.mixup_loss = False

    def sample_negative(self):
        try:
            neg_data = next(self.bg_iter)
        except StopIteration:
            self.bg_iter = iter(self.bg_dataloader)  # Reset the data loader
            neg_data = next(self.bg_iter)
        return neg_data[0]

    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            if not 'blocks.11.mlp.fc2' in name:
                param.requires_grad = False

    def compute_loss(self, pos_feat1, pos_feat2, neg_feat1, neg_feat2, margin=0.5):
        positive_sim = torch.nn.functional.cosine_similarity(pos_feat1, pos_feat2, dim=-1)
        negative_sim = torch.nn.functional.cosine_similarity(neg_feat1, neg_feat2, dim=-1)
        
        tmp = positive_sim + (margin - negative_sim).clamp(min=0)
        return torch.mean(tmp)
    
    def train(self):
        for i in range(self.args.num_epochs):
            # print(self.training_data.pos_classes[:50])
            for batch_ndx, sample in enumerate(self.train_dataloader):
                pos, pos_labels = sample
                pos_labels = torch.cat(pos_labels)
                # print(len(pos), (pos[0]-pos[1]).sum(), pos_labels)
                pos = torch.stack(pos).view(-1, *pos[0][0].shape)
                pos = pos.to(self.args.device, non_blocking=True)
                pos_feat = self.model(pos)
                all_embeddings = pos_feat
                all_labels = pos_labels
                
                # sample negative
                if self.negative_sample > 0:
                    neg = self.sample_negative()
                    neg = torch.stack(neg).view(-1, *neg[0][0].shape)
                    neg = neg.to(self.args.device, non_blocking=True)
                    neg_feat = self.model(neg)
                    
                    if self.mixup_sampling:
                        # randTensor = torch.rand(self.negative_sample).to(pos_feat.device).unsqueeze(1)
                        # randTensor = torch.rand(1).to(pos_feat.device)
                        
                        # negative + mixup
                        # mix_feat = (1-randTensor)*pos_feat[:self.negative_sample] + randTensor*neg_feat
                        # neg_feat = torch.cat((neg_feat, mix_feat), dim=0)
                        # neg_labels = torch.tensor(
                        #     [self.negative_class_label] * (self.negative_sample * 2), dtype=pos_labels.dtype,
                        #     device=pos_labels.device)
                        
                        # replace negative with mixup
                        # print(randTensor.shape, .shape, neg_feat.shape)
                        randTensor = (1.0 - 0.9) * torch.rand(1).to(pos_feat.device).unsqueeze(1) + 0.9
                        pos_tmp = pos_feat[:self.negative_sample].clone()
                        mix_feat = (1-randTensor)*pos_tmp + randTensor*neg_feat
                        neg_feat = mix_feat
                        neg_labels = torch.tensor(
                            [self.negative_class_label] * self.negative_sample, dtype=pos_labels.dtype,
                            device=pos_labels.device)
                        
                    else:
                        neg_labels = torch.tensor(
                            [self.negative_class_label] * self.negative_sample, dtype=pos_labels.dtype,
                            device=pos_labels.device)
                    
                    all_embeddings = torch.cat((pos_feat, neg_feat), dim=0)
                    all_labels = torch.cat((pos_labels, neg_labels), dim=0)
                
                positive_pairs, negative_pairs = self.contrastive_sampler.get_pairs(all_embeddings, all_labels)
                pos_1_feat = all_embeddings[positive_pairs[:, 0]]
                pos_2_feat = all_embeddings[positive_pairs[:, 1]]
                neg_1_feat = all_embeddings[negative_pairs[:, 0]]
                neg_2_feat = all_embeddings[negative_pairs[:, 1]]
                
                self.optimizer.zero_grad()
                loss = self.compute_loss(pos_1_feat, pos_2_feat, neg_1_feat, neg_2_feat)
                print("epoch: {}, batch: {}, loss: {}".format(i, batch_ndx, loss))
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.empty_cache()
            # self.scheduler.step()
            
            if (i+1) % self.args.save_epochs == 0:
                output_f = "epoch{}.pt".format(i+1)
                self.save_weights(i+1, os.path.join(self.args.weights_root, output_f))

    def save_weights(self, epoch, output_f):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, output_f)

    def load_weights(self, f_p):
        checkpoint = torch.load(f_p)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self):
        self.model.eval()
        object_feat = []
        proposal_feat = []
        with torch.no_grad():
            print("computing object features!!")
            for batch_sample in self.eval_object_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)
                object_feat.append(batch_feat.detach().cpu())

            object_feat = torch.cat(object_feat, dim=0)
            print("object_feat1: ", len(object_feat))
            if self.eval_render_dataloader is not None:
                print("computing rendered features!!")
                render_feat = []
                for batch_sample in self.eval_render_dataloader:
                    batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                    render_feat.append(batch_feat.detach().cpu())
                render_feat = torch.cat(render_feat, dim=0)
                # concatenate render feat witg object feat
                object_feat = object_feat.reshape(self.num_class, len(object_feat)//self.num_class, -1)
                render_feat = render_feat.reshape(self.num_class, len(render_feat)//self.num_class, -1)
                
                object_feat = torch.cat((object_feat, render_feat), dim=1).reshape(-1, render_feat.shape[-1])
            
            print("object_feat: ", len(object_feat))
            
            print("computing proposal features!!")
            start_time = time.time()
            for batch_sample in self.eval_proposal_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)
                proposal_feat.append(batch_feat.detach().cpu())
            
            proposal_feat = torch.cat(proposal_feat, dim=0)
            print("proposal_feat: ", len(proposal_feat))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed Time on extracting proposal features:{elapsed_time}")
            
            return object_feat, proposal_feat
    
    # def eval_custom_feat(self, input_feat):
    #     self.model.eval()
    #     with torch.no_grad():
    #         return self.model(input_feat)

class TrainerWithTriplet(nn.Module):
    """
    Finetuning model with Triplet Loss With Last 1 linear layer
    """
    def __init__(self, args, model, object_dataset, proposal_dataset, bg_dataset=None, render_dataset=None, num_workers=4) -> None:
        super(TrainerWithTriplet, self).__init__()
        # args is dictionary type here
        self.args = args
        self.model = model.to(args.device)
        self.model.train()
        self.freeze_layers()
        self.num_class = len(object_dataset.cfg['obj_name'])
        

        num_profile_example = len(object_dataset.samples) // self.num_class
        if render_dataset is not None:
            num_render_example = len(render_dataset.samples) // self.num_class
        else:
            num_render_example = None

        self.training_data = CustomTripletDataset(
            object_dataset, num_class=self.num_class, num_profile_example=num_profile_example,
            render_dataset=render_dataset, num_render_example=num_render_example)
  
        self.train_dataloader = DataLoader(
            self.training_data, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True,
            )
        
        #print("number of training examples: ", len(self.training_data) * args.num_samples)
        
        # ================================================================================
        self.eval_object_dataloader = DataLoader(
            object_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        self.eval_proposal_dataloader = DataLoader(
            proposal_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        if render_dataset is not None:       
            self.eval_render_dataloader = DataLoader(
                render_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                num_workers=num_workers, pin_memory=True,
                )
        else:
            self.eval_render_dataloader = None
        
        
        # 0.1 so far works the best
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.005)
        
        # total number of bg based
        # self.negative_portion = 0.01
        # self.negative_sample = int(bg_feat.shape[0] * self.negative_portion)
        
        # batch size based
        if bg_dataset is not None:
            self.negative_portion = 0.25
            self.negative_sample = int(args.batch_size * args.num_samples * self.negative_portion)
            print("Number of bg examples: {}".format(self.negative_sample))
        
            self.bg_dataloader = DataLoader(
                bg_dataset, batch_size=self.negative_sample, shuffle=True, drop_last=True, 
                num_workers=num_workers, pin_memory=True,
                )
            self.bg_iter = iter(self.bg_dataloader)
            # self.bg_dataset = bg_dataset
            
            self.negative_class_label = self.num_class
        else:
            self.negative_sample = 0
        
        self.triplet_sampler = FunctionNegativeTripletSelector(margin=0.5, negative_selection_fn=hardest_negative) 
        # self.triplet_sampler = FunctionNegativeTripletSelector(margin=0.5, negative_selection_fn=semihard_negative, semi_hard=True)
        # self.triplet_sampler = FunctionNegativeTripletSelector(margin=0.5, negative_selection_fn=random_hard_negative)
        # self.triplet_sampler = AllTripletSelector()
        
        self.mixup_sampling = False
        self.mixup_ratio = 0.5
        self.mixup_loss = False

    def sample_negative(self):
        try:
            neg_data = next(self.bg_iter)
        except StopIteration:
            self.bg_iter = iter(self.bg_dataloader)  # Reset the data loader
            neg_data = next(self.bg_iter)
        return neg_data[0]

    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            if not 'blocks.11.mlp.fc2' in name:
                param.requires_grad = False

    def compute_loss(self, pos_feat1, pos_feat2, neg_feat, margin=0.5):
        # within class distance should be smaller than between class distance
        within_class_sim = torch.nn.functional.cosine_similarity(pos_feat1, pos_feat2, dim=-1)
        between_class_sim = torch.nn.functional.cosine_similarity(pos_feat1, neg_feat, dim=-1)
        
        # between-class distance should be larger than within-class distance by a margin alpha 
        # loss = max(between_class_dist - within_class_dist - alpha, 0)  --> minimize loss, i.e., min (max(....))

        # within class similarity should be larger than between-class similarity by a margin alpha
        # loss = max(between_class_sim  - within_class_sim + alpha, 0)
        tmp = (between_class_sim - within_class_sim + margin).clamp(min=0)
        return torch.mean(tmp)
    
    def train(self):
        for i in range(self.args.num_epochs):
            # print(self.training_data.pos_classes[:50])
            for batch_ndx, sample in enumerate(self.train_dataloader):
                pos, pos_labels = sample
                pos_labels = torch.cat(pos_labels)
                # print(len(pos), (pos[0]-pos[1]).sum(), pos_labels)
                pos = torch.stack(pos).view(-1, *pos[0][0].shape)
                pos = pos.to(self.args.device, non_blocking=True)
                pos_feat = self.model(pos)
                all_embeddings = pos_feat
                all_labels = pos_labels
                
                # sample negative
                if self.negative_sample > 0:
                    neg = self.sample_negative()
                    neg = torch.stack(neg).view(-1, *neg[0][0].shape)
                    neg = neg.to(self.args.device, non_blocking=True)
                    neg_feat = self.model(neg)
                    
                    if self.mixup_sampling:
                        # randTensor = torch.rand(self.negative_sample).to(pos_feat.device).unsqueeze(1)
                        # randTensor = torch.rand(1).to(pos_feat.device)
                        
                        # negative + mixup
                        # mix_feat = (1-randTensor)*pos_feat[:self.negative_sample] + randTensor*neg_feat
                        # neg_feat = torch.cat((neg_feat, mix_feat), dim=0)
                        # neg_labels = torch.tensor(
                        #     [self.negative_class_label] * (self.negative_sample * 2), dtype=pos_labels.dtype,
                        #     device=pos_labels.device)
                        
                        # replace negative with mixup
                        # print(randTensor.shape, .shape, neg_feat.shape)
                        randTensor = (1.0 - 0.9) * torch.rand(1).to(pos_feat.device).unsqueeze(1) + 0.9
                        pos_tmp = pos_feat[:self.negative_sample].clone()
                        mix_feat = (1-randTensor)*pos_tmp + randTensor*neg_feat
                        neg_feat = mix_feat
                        neg_labels = torch.tensor(
                            [self.negative_class_label] * self.negative_sample, dtype=pos_labels.dtype,
                            device=pos_labels.device)
                        
                    else:
                        neg_labels = torch.tensor(
                            [self.negative_class_label] * self.negative_sample, dtype=pos_labels.dtype,
                            device=pos_labels.device)
                    
                    all_embeddings = torch.cat((pos_feat, neg_feat), dim=0)
                    all_labels = torch.cat((pos_labels, neg_labels), dim=0)
                
                triplets = self.triplet_sampler.get_triplets(all_embeddings, all_labels)
                # print("triplets: ", triplets.shape)
                # print(torch.unique(all_labels[triplets[:, 0]]))
                
                pos_1_feat = all_embeddings[triplets[:, 0]]
                pos_2_feat = all_embeddings[triplets[:, 1]]
                neg_feat = all_embeddings[triplets[:, 2]]
                
                self.optimizer.zero_grad()
                loss = self.compute_loss(pos_1_feat, pos_2_feat, neg_feat)
                print("epoch: {}, batch: {}, loss: {}".format(i, batch_ndx, loss))
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.empty_cache()
            # self.scheduler.step()
            
            if (i+1) % self.args.save_epochs == 0:
                output_f = "epoch{}.pt".format(i+1)
                self.save_weights(i+1, os.path.join(self.args.weights_root, output_f))

    def save_weights(self, epoch, output_f):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, output_f)

    def load_weights(self, f_p):
        checkpoint = torch.load(f_p)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self):
        self.model.eval()
        object_feat = []
        proposal_feat = []
        with torch.no_grad():
            print("computing object features!!")
            for batch_sample in self.eval_object_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)
                object_feat.append(batch_feat.detach())

            object_feat = torch.cat(object_feat, dim=0)
            print("object_feat1: ", len(object_feat))
            if self.eval_render_dataloader is not None:
                print("computing rendered features!!")
                render_feat = []
                for batch_sample in self.eval_render_dataloader:
                    batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                    render_feat.append(batch_feat.detach())
                render_feat = torch.cat(render_feat, dim=0)
                # concatenate render feat witg object feat
                object_feat = object_feat.reshape(self.num_class, len(object_feat)//self.num_class, -1)
                render_feat = render_feat.reshape(self.num_class, len(render_feat)//self.num_class, -1)
                
                object_feat = torch.cat((object_feat, render_feat), dim=1).reshape(-1, render_feat.shape[-1])
            
            print("object_feat: ", len(object_feat))
            
            print("computing proposal features!!")
            
            
            for batch_sample in self.eval_proposal_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                
                start_time = time.time()
                
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed Time on proposal feature extraction:{elapsed_time}")                

                proposal_feat.append(batch_feat.detach())
            
            proposal_feat = torch.cat(proposal_feat, dim=0)
            print("proposal_feat: ", len(proposal_feat))

            return object_feat, proposal_feat
    
    # def eval_custom_feat(self, input_feat):
    #     self.model.eval()
    #     with torch.no_grad():
    #         return self.model(input_feat)


class TrainerWithTripletV2(nn.Module):
    """
    Finetuning model with Triplet Loss With Last 2 linear layers
    """
    def __init__(self, args, model, object_dataset, proposal_dataset, bg_dataset=None, render_dataset=None, num_workers=4) -> None:
        super(TrainerWithTripletV2, self).__init__()
        # args is dictionary type here
        self.args = args
        self.model = model.to(args.device)
        self.model.train()
        self.freeze_layers()
        self.num_class = len(object_dataset.cfg['obj_name'])
        

        num_profile_example = len(object_dataset.samples) // self.num_class
        if render_dataset is not None:
            num_render_example = len(render_dataset.samples) // self.num_class
        else:
            num_render_example = None

        self.training_data = CustomTripletDataset(
            object_dataset, num_class=self.num_class, num_profile_example=num_profile_example,
            render_dataset=render_dataset, num_render_example=num_render_example)
  
        self.train_dataloader = DataLoader(
            self.training_data, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True,
            )
        
        #print("number of training examples: ", len(self.training_data) * args.num_samples)
        
        # ================================================================================
        self.eval_object_dataloader = DataLoader(
            object_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        self.eval_proposal_dataloader = DataLoader(
            proposal_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            )
        if render_dataset is not None:       
            self.eval_render_dataloader = DataLoader(
                render_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                num_workers=num_workers, pin_memory=True,
                )
        else:
            self.eval_render_dataloader = None
        
        
        # 0.1 so far works the best
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.5)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.005)
        
        # total number of bg based
        # self.negative_portion = 0.01
        # self.negative_sample = int(bg_feat.shape[0] * self.negative_portion)
        
        # batch size based
        if bg_dataset is not None:
            self.negative_portion = 0.25
            self.negative_sample = int(args.batch_size * args.num_samples * self.negative_portion)
            print("Number of bg examples: {}".format(self.negative_sample))
        
            self.bg_dataloader = DataLoader(
                bg_dataset, batch_size=self.negative_sample, shuffle=True, drop_last=True, 
                num_workers=num_workers, pin_memory=True,
                )
            self.bg_iter = iter(self.bg_dataloader)
            # self.bg_dataset = bg_dataset
            
            self.negative_class_label = self.num_class
        else:
            self.negative_sample = 0
        
        self.triplet_sampler = FunctionNegativeTripletSelector(margin=0.5, negative_selection_fn=hardest_negative) 
        # self.triplet_sampler = FunctionNegativeTripletSelector(margin=0.5, negative_selection_fn=semihard_negative, semi_hard=True)
        # self.triplet_sampler = FunctionNegativeTripletSelector(margin=0.5, negative_selection_fn=random_hard_negative)
        # self.triplet_sampler = AllTripletSelector()
        
        self.mixup_sampling = False
        self.mixup_ratio = 0.5
        self.mixup_loss = False

    def sample_negative(self):
        try:
            neg_data = next(self.bg_iter)
        except StopIteration:
            self.bg_iter = iter(self.bg_dataloader)  # Reset the data loader
            neg_data = next(self.bg_iter)
        return neg_data[0]

    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            if not 'blocks.11.mlp.fc' in name:
                param.requires_grad = False

    def compute_loss(self, pos_feat1, pos_feat2, neg_feat, margin=0.5):
        # within class distance should be smaller than between class distance
        within_class_sim = torch.nn.functional.cosine_similarity(pos_feat1, pos_feat2, dim=-1)
        between_class_sim = torch.nn.functional.cosine_similarity(pos_feat1, neg_feat, dim=-1)
        
        # between-class distance should be larger than within-class distance by a margin alpha 
        # loss = max(between_class_dist - within_class_dist - alpha, 0)  --> minimize loss, i.e., min (max(....))

        # within class similarity should be larger than between-class similarity by a margin alpha
        # loss = max(between_class_sim  - within_class_sim + alpha, 0)
        tmp = (between_class_sim - within_class_sim + margin).clamp(min=0)
        return torch.mean(tmp)
    
    def train(self):
        for i in range(self.args.num_epochs):
            # print(self.training_data.pos_classes[:50])
            for batch_ndx, sample in enumerate(self.train_dataloader):
                pos, pos_labels = sample
                pos_labels = torch.cat(pos_labels)
                # print(len(pos), (pos[0]-pos[1]).sum(), pos_labels)
                pos = torch.stack(pos).view(-1, *pos[0][0].shape)
                pos = pos.to(self.args.device, non_blocking=True)
                pos_feat = self.model(pos)
                all_embeddings = pos_feat
                all_labels = pos_labels
                
                # sample negative
                if self.negative_sample > 0:
                    neg = self.sample_negative()
                    neg = torch.stack(neg).view(-1, *neg[0][0].shape)
                    neg = neg.to(self.args.device, non_blocking=True)
                    neg_feat = self.model(neg)
                    
                    if self.mixup_sampling:
                        # randTensor = torch.rand(self.negative_sample).to(pos_feat.device).unsqueeze(1)
                        # randTensor = torch.rand(1).to(pos_feat.device)
                        
                        # negative + mixup
                        # mix_feat = (1-randTensor)*pos_feat[:self.negative_sample] + randTensor*neg_feat
                        # neg_feat = torch.cat((neg_feat, mix_feat), dim=0)
                        # neg_labels = torch.tensor(
                        #     [self.negative_class_label] * (self.negative_sample * 2), dtype=pos_labels.dtype,
                        #     device=pos_labels.device)
                        
                        # replace negative with mixup
                        # print(randTensor.shape, .shape, neg_feat.shape)
                        randTensor = (1.0 - 0.9) * torch.rand(1).to(pos_feat.device).unsqueeze(1) + 0.9
                        pos_tmp = pos_feat[:self.negative_sample].clone()
                        mix_feat = (1-randTensor)*pos_tmp + randTensor*neg_feat
                        neg_feat = mix_feat
                        neg_labels = torch.tensor(
                            [self.negative_class_label] * self.negative_sample, dtype=pos_labels.dtype,
                            device=pos_labels.device)
                        
                    else:
                        neg_labels = torch.tensor(
                            [self.negative_class_label] * self.negative_sample, dtype=pos_labels.dtype,
                            device=pos_labels.device)
                    
                    all_embeddings = torch.cat((pos_feat, neg_feat), dim=0)
                    all_labels = torch.cat((pos_labels, neg_labels), dim=0)
                
                triplets = self.triplet_sampler.get_triplets(all_embeddings, all_labels)
                # print("triplets: ", triplets.shape)
                # print(torch.unique(all_labels[triplets[:, 0]]))
                
                pos_1_feat = all_embeddings[triplets[:, 0]]
                pos_2_feat = all_embeddings[triplets[:, 1]]
                neg_feat = all_embeddings[triplets[:, 2]]
                
                self.optimizer.zero_grad()
                loss = self.compute_loss(pos_1_feat, pos_2_feat, neg_feat)
                print("epoch: {}, batch: {}, loss: {}".format(i, batch_ndx, loss))
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.empty_cache()
            # self.scheduler.step()
            
            if (i+1) % self.args.save_epochs == 0:
                output_f = "epoch{}.pt".format(i+1)
                self.save_weights(i+1, os.path.join(self.args.weights_root, output_f))

    def save_weights(self, epoch, output_f):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, output_f)

    def load_weights(self, f_p):
        checkpoint = torch.load(f_p)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self):
        self.model.eval()
        object_feat = []
        proposal_feat = []
        with torch.no_grad():
            print("computing object features!!")
            for batch_sample in self.eval_object_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)
                object_feat.append(batch_feat.detach().cpu())

            object_feat = torch.cat(object_feat, dim=0)
            print("object_feat1: ", len(object_feat))
            if self.eval_render_dataloader is not None:
                print("computing rendered features!!")
                render_feat = []
                for batch_sample in self.eval_render_dataloader:
                    batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                    render_feat.append(batch_feat.detach().cpu())
                render_feat = torch.cat(render_feat, dim=0)
                # concatenate render feat witg object feat
                object_feat = object_feat.reshape(self.num_class, len(object_feat)//self.num_class, -1)
                render_feat = render_feat.reshape(self.num_class, len(render_feat)//self.num_class, -1)
                
                object_feat = torch.cat((object_feat, render_feat), dim=1).reshape(-1, render_feat.shape[-1])
            
            print("object_feat: ", len(object_feat))
            
            print("computing proposal features!!")
            for batch_sample in self.eval_proposal_dataloader:
                # print("batch_sample[0]: ", batch_sample[0].shape)
                batch_feat = self.model(batch_sample[0][0].to(self.args.device, non_blocking=True))
                # print("object_feat: ", batch_feat.shape)
                proposal_feat.append(batch_feat.detach().cpu())
            
            proposal_feat = torch.cat(proposal_feat, dim=0)
            print("proposal_feat: ", len(proposal_feat))
            
            return object_feat, proposal_feat
    
    # def eval_custom_feat(self, input_feat):
    #     self.model.eval()
    #     with torch.no_grad():
    #         return self.model(input_feat)
