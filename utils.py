from itertools import combinations

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def filter_bg(neg_feat, pos_feat, 
              threshold=0.25, num_class=100, num_example=24):
    # compute cosine similarity between pos_feat and neg_feat
    res = []
    pos_feat_reshaped = pos_feat.reshape(num_class, num_example, -1)
    for neg_idx in range(len(neg_feat)):
        # print(neg_idx)
        this_neg_feat = neg_feat[neg_idx] # 384
        for pos_class_idx in range(num_class):
            pos_feat_this_class = pos_feat_reshaped[pos_class_idx] # num_example x 384
            cos_sim = torch.nn.functional.cosine_similarity(
                this_neg_feat.unsqueeze(0).unsqueeze(1).expand(-1, num_example, -1), 
                pos_feat_this_class.unsqueeze(0).expand(1, -1, -1), 
                dim=-1)
            # shape (1, num_example)
            
            # max implementation
            # cos_sim_score = torch.max(cos_sim, dim=-1)[0]
            
            # avg implementation
            cos_sim_score = torch.mean(cos_sim, dim=-1)
            if cos_sim_score > threshold:
                # if the proposal has higher simlarity with one object, we keep it
                res.append(neg_idx)
                break
                
    return res

 # ================================================================================= #

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def cosdist(vectors):
    # vectors_1 = vectors.unsqueeze(0).expand(vectors.shape[0], -1, -1)
    # vectors_2 = vectors.unsqueeze(1).expand(-1, vectors.shape[0], -1)
    
    vectors_1 = vectors.unsqueeze(1).expand(-1, vectors.shape[0], -1)
    vectors_2 = vectors.unsqueeze(0).expand(vectors.shape[0], -1, -1)
    distance_matrix = torch.nn.functional.cosine_similarity(vectors_1, vectors_2, dim=-1)
    # print("distance_matrix, ", distance_matrix)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu
        self.ignore_label = 100

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        positive_labels = np.array([l for l in labels if l != self.ignore_label])
        positive_pairs = np.array(list(combinations(range(len(positive_labels)), 2)))
        positive_pairs = positive_pairs[(positive_labels[positive_pairs[:, 0]] == positive_labels[positive_pairs[:, 1]]).nonzero()]

        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()
        self.ignore_label = 100

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            if label == self.ignore_label:
                continue
            
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True, semi_hard=False):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.ignore_label = 100
        self.semi_hard = semi_hard


    def get_triplets(self, embeddings, labels):
        # we consider negatives from other classes and background seperately 
        self.triplet_labels = []
        if self.cpu:
            embeddings = embeddings.cpu()
        # distance_matrix = pdist(embeddings)
        distance_matrix = cosdist(embeddings) # pair-wise cosine simlarity
        
        # print("distance_matrix", distance_matrix.shape)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        # print("all labels: ", set(labels))
        for label in set(labels):
            if label == self.ignore_label:
                continue
            # print(label)
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            # print(len(label_indices))
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)
            # print(len(anchor_positives))

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                # tmp = (between_class_sim - within_class_sim + margin).clamp(min=0)
                # for cosine simlarity implementation only
                loss_values = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] - ap_distance + self.margin
                loss_values = loss_values.clamp(min=0).data.cpu().numpy()
                
                if self.semi_hard:
                    hard_negative = self.negative_selection_fn(loss_values, self.margin) # for semi-hard
                else:
                    hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
            # print("triplets len", len(triplets))
                    self.triplet_labels.append(label)

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

    def get_triplet_labels(self):
        return np.array(self.triplet_labels)


#  ==========================

def fun_CosSim(Mat_A, Mat_B, norm=1, ):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    D = Mat_A.mm(torch.t(Mat_B))
    D.fill_diagonal_(-norm)
    return D

def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff
    
class SCTLoss(nn.Module):
    def __init__(self, method, lam=1):
        super(SCTLoss, self).__init__()
        
        if method=='sct':
            self.sct = True
            self.semi = False
        elif method=='hn':
            self.sct = False
            self.semi = False
        elif method=='shn':
            self.sct = False
            self.semi = True
        else:
            print('loss type is not supported')
            
        self.lam = lam

    def forward(self, fvec, Lvec):
        # number of images
        N = Lvec.size(0)
        
        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        
        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix
        CosSim = fun_CosSim(fvec_norm, fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = CosSim.clone().detach()
        D_detach_P[Diff] = -1
        D_detach_P[D_detach_P>0.9999] = -1
        V_pos, I_pos = D_detach_P.max(1)
 
        # valid positive pairs(prevent pairs with duplicated images)
        Mask_pos_valid = (V_pos>-1)&(V_pos<1)

        # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos]
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1
        
        # Masking out non-Semi-Hard Negative
        if self.semi:    
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff] = -1
            
        V_neg, I_neg = D_detach_N.max(1)
            
        # valid negative pairs
        Mask_neg_valid = (V_neg>-1)&(V_neg<1)

        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()
        
        # Mask all valid triplets
        Mask_valid = Mask_pos_valid&Mask_neg_valid
        
        # Mask hard/easy triplets
        HardTripletMask = ((Neg>Pos) | (Neg>0.8)) & Mask_valid
        EasyTripletMask = ((Neg<Pos) & (Neg<0.8)) & Mask_valid
        
        # number of hard triplet
        hn_ratio = (Neg>Pos)[Mask_valid].clone().float().mean().cpu()
        
        # triplets
        Triplet_val = torch.stack([Pos,Neg],1)
        Triplet_idx = torch.stack([I_pos,I_neg],1)
        
        Triplet_val_log = Triplet_val.clone().detach().cpu()
        Triplet_idx_log = Triplet_idx.clone().detach().cpu()
        
        # loss
        if self.sct: # SCT setting
            
            loss_hardtriplet = Neg[HardTripletMask].sum()
            loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask,:]/0.1, dim=1)[:,0].sum()
            
            N_hard = HardTripletMask.float().sum()
            N_easy = EasyTripletMask.float().sum()
            
            if torch.isnan(loss_hardtriplet) or N_hard==0:
                loss_hardtriplet, N_hard = 0, 0
                print('No hard triplets in the batch')
                
            if torch.isnan(loss_easytriplet) or N_easy==0:
                loss_easytriplet, N_easy = 0, 0
                print('No easy triplets in the batch')
                
            N = N_easy + N_hard
            if N==0: N=1
            loss = (loss_easytriplet + self.lam*loss_hardtriplet)/N
                
        else: # Standard Triplet Loss setting
            
            loss = -F.log_softmax(Triplet_val[Mask_valid,:]/0.1, dim=1)[:,0].mean()
            
        print('loss:{:.3f} hn_rt:{:.3f}'.format(loss.item(), hn_ratio.item()), end='\r')

        return loss, Triplet_val_log, Triplet_idx_log, hn_ratio