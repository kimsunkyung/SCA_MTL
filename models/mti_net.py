#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

""" 
    MTI-Net implementation based on HRNet backbone 
    https://arxiv.org/pdf/2001.06902.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock
from models.layers import SEBlock
from models.padnet import MultiTaskDistillationModule
from models.seg_hrnet import SwinTransformer, GlobalContextAttentionBlock


class InitialTaskPredictionModule(nn.Module):
    """ Module to make the inital task predictions """

    def __init__(self, p, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPredictionModule, self).__init__()
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict(
                {task: nn.Sequential(BasicBlock(channels, channels), BasicBlock(channels, channels)) for task in
                 self.auxilary_tasks})

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False),
                                           nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(BasicBlock(input_channels, task_channels, downsample=downsample),
                                              BasicBlock(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict(
            {task: nn.Conv2d(task_channels, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1) for task in self.auxilary_tasks})

    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None:  # Concat features that were propagated from previous scale
            x = {t: torch.cat(
                (features_curr_scale, F.interpolate(features_prev_scale[t], scale_factor=2, mode='bilinear')), 1) for t
                 in self.auxilary_tasks}

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' % (t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' % (t)])

        return out



class FPM(nn.Module):
    """ Feature Propagation Module """

    def __init__(self, tasks, auxilary_tasks, per_task_channels):
        super(FPM, self).__init__()
        # General
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N * per_task_channels)

        # Non-linear function f
        self.swtf = SwinTransformer(224, 1, self.shared_channels, self.shared_channels, [1], [4], 7,
                                    4, True, None, 0,
                                    0, 0.2, nn.LayerNorm, False, True, [0], -1, False)

        # Dimensionality reduction
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                   nn.BatchNorm2d(self.per_task_channels))

        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                   downsample=downsample)

        # SEBlock
        self.se = nn.ModuleDict({task: SEBlock(self.per_task_channels) for task in self.auxilary_tasks})


    def forward(self, x):
        # Get shared representation
        concat = torch.cat([x[(task)] for task in self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.swtf(concat)
        shared = shared[0]
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)  # Per task attention mask
        shared = torch.mul(mask, concat.view(B, C // self.N, self.N, H, W)).view(B, -1, H, W)

        # Perform dimensionality reduction
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x[(task)]

        return out





class MTINet(nn.Module):
    """
        MTI-Net implementation based on HRNet backbone
        https://arxiv.org/pdf/2001.06902.pdf
    """

    def __init__(self, p, backbone, backbone_channels, heads):
        super(MTINet, self).__init__()
        # General
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels

        # Backbone
        self.backbone = backbone


        # Encoder Transformer
        self.swTransform1 = SwinTransformer(224, 1, 18, 18, [2], [6], 7, 4, True, None, 0,
                                           0, 0.2, nn.LayerNorm, False, True, [0], -1, False)

        self.swTransform2 = SwinTransformer(224, 1, 36, 36, [2], [6], 7, 4, True, None, 0,
                                            0, 0.2, nn.LayerNorm, False, True, [0], -1, False)

        self.swTransform3 = SwinTransformer(224, 1, 72, 72, [2], [6], 7, 4, True, None, 0,
                                            0, 0.2, nn.LayerNorm, False, True, [0], -1, False)

        self.swTransform4 = SwinTransformer(224, 1, 144, 144, [2], [6], 7, 4, True, None, 0,
                                            0, 0.2, nn.LayerNorm, False, True, [0], -1, False)

        # Decoder Transformer
        # Feature Propagation Module
        self.fpm_scale_3 = FPM(self.tasks, self.auxilary_tasks, self.channels[3])
        self.fpm_scale_2 = FPM(self.tasks, self.auxilary_tasks, self.channels[2])
        self.fpm_scale_1 = FPM(self.tasks, self.auxilary_tasks, self.channels[1])

        # Initial task predictions at multiple scales
        self.scale_0 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[0] + self.channels[1],
                                                   self.channels[0])
        self.scale_1 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[1] + self.channels[2],
                                                   self.channels[1])
        self.scale_2 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[2] + self.channels[3],
                                                   self.channels[2])
        self.scale_3 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[3], self.channels[3])

        # Distillation at multiple scales
        self.distillation_scale_0 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[0])
        self.distillation_scale_1 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[1])
        self.distillation_scale_2 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[2])
        self.distillation_scale_3 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[3])


        # Feature aggregation through HRNet heads
        self.heads = heads

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # Backbone
        x = self.backbone(x)

        encoder_tf0 = self.swTransform1(x[0])[0]
        encoder_tf1 = self.swTransform2(x[1])[0]
        encoder_tf2 = self.swTransform3(x[2])[0]
        encoder_tf3 = self.swTransform4(x[3])[0]

        x[3] = torch.cat([encoder_tf3, x[3]], dim=1)
        x[2] = torch.cat([encoder_tf2, x[2]], dim=1)
        x[1] = torch.cat([encoder_tf1, x[1]], dim=1)
        x[0] = torch.cat([encoder_tf0, x[0]], dim=1)

        # Predictions at multiple scales
        # Scale 3
        x_3 = self.scale_3(x[3])
        features_3 = self.distillation_scale_3(x_3)
        x_3_fpm = self.fpm_scale_3(features_3)
        # Scale 2
        x_2 = self.scale_2(x[2], x_3_fpm)
        features_2 = self.distillation_scale_2(x_2)
        x_2_fpm = self.fpm_scale_2(features_2)
        # Scale 1
        x_1 = self.scale_1(x[1], x_2_fpm)
        features_1 = self.distillation_scale_1(x_1)
        x_1_fpm = self.fpm_scale_1(features_1)
        # Scale 0
        x_0 = self.scale_0(x[0], x_1_fpm)
        features_0 = self.distillation_scale_0(x_0)

        out['deep_supervision'] = {'scale_0': x_0, 'scale_1': x_1, 'scale_2': x_2, 'scale_3': x_3}

        # Distillation + Output
        multi_scale_features = {t: [features_0[t], features_1[t], features_2[t], features_3[t]] for t in self.tasks}

        # Feature aggregation
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](multi_scale_features[t]), img_size, mode='bilinear')

        return out
