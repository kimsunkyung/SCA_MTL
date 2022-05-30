#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations, \
    get_val_dataset, get_train_dataloader, get_val_dataloader, \
    get_optimizer, get_model, adjust_learning_rate, \
    get_criterion
from utils.logger import Logger
from train.train_utils import train_vanilla
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions, \
    eval_all_results
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment', required=True)
parser.add_argument('--config_exp',
                    help='Config file for the experiment', required=True)
parser.add_argument('--pretrained_model',
                    help='pretrained model path', required=True)
args = parser.parse_args()


def main():
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    best_m = torch.load(args.pretrained_model)
    model.load_state_dict(best_m)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))

    # Transforms
    train_transforms, val_transforms = get_transformations(p)
    val_dataset = get_val_dataset(p, val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Val transformations:')
    print(val_transforms)

    # Main loop
    print(colored('Starting main loop', 'blue'))
    save_model_predictions(p, val_dataloader, model)
    curr_result = eval_all_results(p)


if __name__ == "__main__":
    main()
