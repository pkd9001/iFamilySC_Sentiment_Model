# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:25:28 2022

@author: user
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from customdataset import CustomDataset
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
class LabelSmoothingLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    
def dataloader(batch_size, num_workers):
    
    data_ = pd.read_csv('./data/label_data_220511_202205230048.csv')
    data = data_
    data = data.drop(columns=['idx','data_pk','evaluation_price','evaluation_vitality','evaluation_applicability'])
    data = data[data['evaluation'] != -1]

    dataset_train, dataset_test = train_test_split(data, test_size=0.2, random_state=0, stratify=pd.DataFrame(data)['evaluation'])
    
    train_data = CustomDataset(dataset_train)
    test_data = CustomDataset(dataset_test)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)#, collate_fn=None)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)#, collate_fn=None)
    
    return train_loader, test_loader






























