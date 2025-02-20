"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl


import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(prediction, ground_truth):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    l2_loss = nn.MSELoss()
    loss = l2_loss(prediction, ground_truth)

    return loss


class HomographyModelSupervised(pl.LightningModule):
    def __init__(self):
        super(HomographyModelSupervised, self).__init__()
        self.model = SupervisedNet()

    def forward(self, patches):
        return self.model(patches)

    def validation_step(self, img_batch, label_batch):
        pred = self.model(img_batch)
        loss = LossFn(pred, label_batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

class SupervisedNet(pl.LightningModule):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()

        self.seq1 = nn.Sequential(nn.Conv2d(6, 64, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2, 2), stride=2))
        self.seq2 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2, 2), stride=2))
        self.seq3 = nn.Sequential(nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2, 2), stride=2))
        self.seq4 = nn.Sequential(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*16*16, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        """
        Input:
        x is a MiniBatch of the stacked patches (images)
        Outputs:
        out - output of the network
        """
        out = self.seq1(x)
        out = self.seq2(out)
        out = self.seq3(out)
        out = self.seq4(out)
        
        out = out.reshape([x.size(0), -1])
        
        out = self.fc1(out)
        out = F.relu(out)
        
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        return out
