#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import glob
import json
import warnings

import cv2
import torch
import numpy as np
from PIL.Image import Image

from utils.utils import mkdir_if_missing
from losses.loss_functions import BalancedCrossEntropyLoss
from utils.mypath import MyPath, PROJECT_ROOT_DIR

class EdgeMeter(object):
    def __init__(self, pos_weight):
        self.loss = 0
        self.n = 0
        self.loss_function = BalancedCrossEntropyLoss(size_average=True, pos_weight=pos_weight)
        
    @torch.no_grad()
    def update(self, pred, gt):
        gt = gt.squeeze()
        pred = pred.float().squeeze() / 255.
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self, verbose=True):
        eval_dict = {'loss': self.loss / self.n}

        if verbose:
            print('\n Edge Detection Evaluation')
            print('Edge Detection Loss %.3f' %(eval_dict['loss']))

        return eval_dict


def eval_edge(loader, folder):
    total_edge = 0.0
    n_valid = 0.0
    tp = [0] * 1
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating depth: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        pred = np.array(Image.open(filename)).astype(np.float32)

        label = sample['edge']

        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_LINEAR)

        valid_mask = (label != 0)
        n_valid += np.sum(valid_mask)

        label[label == 0] = 1e-9  # Avoid overflow/underflow
        pred[pred <= 0] = 1e-9

        for i_part in range(2):
            tmp_gt = (label == i_part)
            tmp_pred = (pred == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & valid_mask)

    eval_result = dict()
    eval_result['osdf'] = np.sqrt(tp[1] / n_valid)

    return eval_result


def eval_edge_predictions(p, database, save_dir):
    """ The edge are evaluated through seism """

    print('Evaluate the edge prediction using seism ... This can take a while ...')

    if database == 'NYUD':
        from data.nyud import NYUD_MT
        split = 'val'
        db = NYUD_MT(split=split, do_edge=True, overfit=False)
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_edge'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (Edge)')
    eval_results = eval_edge(db, os.path.join(save_dir, 'edge'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results
    print('Results for Edge Estimation')


    return eval_results
