import cv2
import copy
import json
import mmcv
import warnings
import os
import random
import torch
import numpy as np
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def getColor():
    c1 = random.randint(16, 255)
    c2 = random.randint(16, 255)
    c3 = random.randint(16, 255)
    return (c1, c2, c3)

def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_scale=0.5,
                      font_size=13,
                      win_name='',
                      fig_size=(15, 10),
                      show=True,
                      wait_time=0,
                      out_file=None,
                      gt=False):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_scale (float): Font scales of texts. Default: 0.5
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        fig_size (tuple): Figure size of the pyplot figure. Default: (15, 10)
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    warnings.warn('"font_scale" will be deprecated in v2.9.0,'
                  'Please use "font_size"')
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).copy()

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        scores = np.sqrt(scores)
        inds = scores > score_thr
        bboxes = bboxes[inds, :][0:1]
        labels = labels[inds][0:1]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(max(labels) + 1)]
        else:
            # specify color
            mask_colors = [np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)] * (max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    img = np.ascontiguousarray(img)

    vis_img = img.copy()

    plt.figure(win_name, figsize=fig_size)
    plt.title(win_name)
    plt.axis('off')
    ax = plt.gca()

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[label] if class_names is not None else f'Obj. {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        
        if gt:
            x,y = bbox_int[0], bbox_int[1]
        else:
            x,y = bbox_int[0], bbox_int[3]

        if gt:
            ax.text(x,
                    y,
                    f'{label_text}',
                    bbox={
                        'facecolor': 'black',
                        'alpha': 1.0,
                        'pad': 0.7,
                        'edgecolor': 'none'},
                    color=text_color,
                    fontsize=font_size,
                    verticalalignment='top',
                    horizontalalignment='left')
        else:
            ax.text(x,
                    y,
                    f'{label_text}',
                    bbox={
                        'facecolor': 'white',
                        'alpha': 1.0,
                        'pad': 0.7,
                        'edgecolor': 'none'},
                    color=text_color,
                    fontsize=font_size,
                    verticalalignment='top',
                    horizontalalignment='left')

        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if i==0:
        #     vis_img = cv2.rectangle(vis_img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (36,255,12), thickness)
        #     vis_img = cv2.putText(vis_img, f'{label_text}', (bbox_int[0], bbox_int[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), thickness)
    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    if out_file is not None:
        dir_name = os.path.abspath(os.path.dirname(out_file))
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(out_file)
        if not show:
            plt.close()
    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
            plt.close()
    return mmcv.rgb2bgr(vis_img)


def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:     # save one box
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # save box[i] with max score
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # idx:[N-1,] order:[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]
    return torch.LongTensor(keep)

def batched_nms(bboxes, scores, classes, iou_thresh=0.0):
    """
    perform nms per class.
    ref: torchvision.ops.batch_nms()
    """
    keep = []
    for class_id in torch.unique(classes):
        curr_id = torch.where(classes == class_id)[0]
        curr_keep = nms(bboxes[curr_id], scores[curr_id], threshold=iou_thresh)
        keep.extend(curr_id[curr_keep].tolist())

    return torch.LongTensor(keep)

def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0.0,
                         gt_bbox_color='green',
                         gt_text_color='green',
                         gt_mask_color='green',
                         det_bbox_color=(255, 102, 61),
                         det_text_color=(255, 102, 61),
                         det_mask_color=(255, 102, 61),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         fig_size=(15, 10),
                         show=False,
                         wait_time=0,
                         out_file=None):
    
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    gt_masks = annotation.get('gt_masks', None)

    assert 'bboxes' in result
    assert 'labels' in result
    segms = result.get('masks', None)

    img = mmcv.imread(img)

    # only visualize Prediction overlapped on GT annotations
    keep = []
    for idx, _ in enumerate(annotation['gt_labels']):
        gt_bbox = annotation['gt_bboxes'][idx]
        gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

        dt_bbox = result['bboxes'].copy()
        dt_area = (dt_bbox[:,2] - dt_bbox[:,0]) * (dt_bbox[:,3] - dt_bbox[:,1])
        xx1 = np.maximum(gt_bbox[0], dt_bbox[:,0])
        yy1 = np.maximum(gt_bbox[1], dt_bbox[:,1])
        xx2 = np.minimum(gt_bbox[2], dt_bbox[:,2])
        yy2 = np.minimum(gt_bbox[3], dt_bbox[:,3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (gt_area + dt_area - inter)
        ind = list(ovr).index(np.max(ovr))
        if np.max(ovr) > 0 and (ind not in keep):
            keep.append(ind)
    print(keep)

    img = imshow_det_bboxes(
        img,
        result['bboxes'][keep],
        result['labels'][keep],
        segms=None,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color=det_bbox_color,
        text_color=det_text_color,
        mask_color=det_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        fig_size=fig_size,
        show=False,
        gt=False)
    
    img = imshow_det_bboxes(
        img,
        annotation['gt_bboxes'],
        annotation['gt_labels'],
        segms=None,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        fig_size=fig_size,
        show=show,
        wait_time=wait_time,
        out_file=out_file,
        gt=True)
    return img 