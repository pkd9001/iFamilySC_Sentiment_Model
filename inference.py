# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:50:00 2022

@author: TFG5476H
"""

import torch
import pandas as pd
import numpy as np
from transformers import (AutoTokenizer,
                          AdamW,
                          BertForSequenceClassification
                          )
from customdataset import CustomDataset
from customdataset_test import CustomDataset
from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
pretrain = "klue/bert-base"

model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=3).to(device)

batch_size = 1
num_workers = 0

def predict(predict_sentence):
    
    another_test = CustomDataset(predict_sentence)
    another_dataloader = DataLoader(another_test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    model_state_dict = torch.load("model/Epoch_2_loss_0.0211.pt")
    model.load_state_dict(model_state_dict) 
    
    model.eval()
    
    for batch in another_dataloader:
        batch = tuple(v.to(device) for v in batch)
        input_ids, token_type_ids, attention_masks, labels = batch
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        labels=labels
                        )
            logits = out[1]
    
        test_eval=[]
        for i in logits:
            logits=i
            logits = logits.detach().cpu().numpy()
    
            if np.argmax(logits) == 0:
                test_eval.append("neutral")
            elif np.argmax(logits) == 1:
                test_eval.append("negative")
            elif np.argmax(logits) == 2:
                test_eval.append("positive")
    
        print(test_eval[0])
        print(logits)

if __name__ == "__main__":
    end = 1
    while end == 1 :
        print("test end -> done")
        test = input("text: ")
        if test == 'done' :
          break
        else:
          another = pd.DataFrame({'data_text': [test], 'evaluation': [0]})
          predict(another)
        print("\n")