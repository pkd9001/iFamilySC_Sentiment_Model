# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:29:47 2022

@author: user
"""

import torch
import torch.nn as nn


from transformers import (AdamW,
                          BertForSequenceClassification
                          )
from transformers.optimization import get_linear_schedule_with_warmup

import numpy as np
import pandas as pd

from tqdm import tqdm

from custom_utils import *

def main():
    seed_everything(42)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    pretrain = "klue/bert-base"
    
    EPOCHS = 10
    batch_size = 32
    warmup_ratio = 0.1
    max_grad_norm = 1
    max_length = 512
    num_workers = 0
    smoothing = 0.1
    
    train_loader, test_loader = dataloader(batch_size, num_workers)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_step = int(total_steps * warmup_ratio)
    
    model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=3,
                                                          # output_attentions=True,
                                                          # output_hidden_states = True,
                                                          # attention_probs_dropout_prob=0.5,
                                                          # hidden_dropout_prob=0.5,
                                                          ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)
    loss_ls = LabelSmoothingLoss(smoothing)
        
    PATH = './model/'
    
    for i in range(EPOCHS):
        total_loss = 0.0
        correct_train = 0
        correct_eval = 0
        total_train = 0
        total_eval = 0
        batches = 0
        f1_mac_ = 0
        recall_mac_ = 0
        precision_mac_ = 0
        f1_mic_ = 0
        recall_mic_ = 0
        precision_mic_ = 0
    
        model.train()
        for batch in tqdm(train_loader):
            batch = tuple(v.to(device) for v in batch)
            input_ids, token_type_ids, attention_masks, labels = batch
            
            out = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        labels=labels)
            logits = out[1]
            
            loss = (loss_ls(logits, labels))**2
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
              
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == labels).sum()
            total_train += len(labels)
    
        print("")
        print("epoch {} Train Loss {:.6f} train acc {:.6f}".format(i+1,
                                                           torch.true_divide(total_loss, total_train),
                                                           torch.true_divide(correct_train, total_train)))
        print("")
    
        torch.save(model.state_dict(), PATH + 'Epoch_{}_loss_{:.4f}.pt'.format(i+1, torch.true_divide(total_loss, total_train)))
        
        model.eval()
        for batch in tqdm(test_loader):
            batch = tuple(v.to(device) for v in batch)
            input_ids, token_type_ids, attention_masks, labels = batch
            with torch.no_grad():
                out = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_masks,
                            labels=labels
                            )
                logits = out[1]
                _, predicted = torch.max(logits, 1)
                correct_eval += (predicted == labels).sum()
                total_eval += len(labels)
            
        print("epoch :{} test acc :{:.6f}".format(i+1, torch.true_divide(correct_eval, total_eval)))
        print("")
        print("#######end epoch:",i+1,"#######")
        print("")

if __name__ == "__main__":
	main()











































