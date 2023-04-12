# -*- coding: utf-8 -*-
"""
Created on Mon May 23 01:33:58 2022

@author: user
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, Recall, Precision
from sklearn.metrics import f1_score, precision_score, recall_score


from transformers import (AutoTokenizer,
                          AutoModelForMaskedLM,
                          AdamW,
                          AutoModelForSequenceClassification,
                          DataCollatorForLanguageModeling,
                          BertForSequenceClassification
                          )
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
from gluonnlp.data import SentencepieceTokenizer
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE

from tqdm import tqdm
import re

import os
import random
import math
from EDA import EDA as eda

#%%

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%%

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
pretrain = "klue/bert-base"

#%%

EPOCHS = 1
batch_size = 2
warmup_ratio = 0.1
max_grad_norm = 1
max_length = 32
num_workers = 0

#%%
data_ = pd.read_csv('C:/Users/user/Desktop/data/if/checkpoint3/data/label_data/label_data_220511_202205230048.csv')

data = data_

data = data.drop(columns=['idx','data_pk','evaluation_price','evaluation_vitality','evaluation_applicability'])
data = data[data['evaluation'] != -1]

dataset_train, dataset_test = train_test_split(data, test_size=0.2, random_state=0, stratify=pd.DataFrame(data)['evaluation'])

#%%

class CustomDataset:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        dataset = self.dataset.iloc[idx]
        datas = self.tokenizer(dataset['data_text'],#padding='max_length', truncation=True)
                                    padding='max_length',
                                    truncation=True#,
                                    # max_length=max_length,
                                    #return_special_tokens_mask=True
                                    )
        input_ids = torch.tensor(datas['input_ids'])
        token_type_ids = torch.tensor(datas['token_type_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        labels = torch.tensor(self.dataset.iloc[idx]['evaluation'])
        
        return input_ids, token_type_ids, attention_mask, labels
    
#%%

# def resampling_data(df):
#     inputs = tokenizer(df['data_text'].tolist(), return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
#     input_ids = inputs.input_ids
#     token_type_ids = inputs.token_type_ids
#     attention_mask = inputs.attention_mask
#     x = [[input_id,token, mask] for input_id, token, mask in zip(input_ids, token_type_ids, attention_mask)]
#     y = df['evaluation'].tolist()
#     return x, y

# x, y = resampling_data(dataset_train)

# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(x, y)

# smote = BorderlineSMOTE(random_state=42)
# x_ros, y_ros = smote.fit_resample(x, y)

# class ResampledDataset(torch.utils.data.Dataset): 
#     def __init__(self, x_rus, y_rus):
#         self.input_ids = []
#         self.token_type_ids = []
#         self.attention_mask = []
#         for input_id, token, mask in x_rus:
#             self.input_ids.append(input_id)
#             self.token_type_ids.append(token)
#             self.attention_mask.append(mask)
#         self.labels = y_rus

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         input_ids = self.input_ids[idx]
#         token_type_ids = self.token_type_ids[idx]
#         attention_mask = self.attention_mask[idx]
#         labels = torch.tensor(self.labels[idx])
#         return input_ids, token_type_ids, attention_mask, labels

# #%%
# '''oversampling'''

# train_data = ResampledDataset(x_ros, y_ros)
# test_data = CustomDataset(dataset_test)

#%%
# '''EDA'''

# data_2 = dataset_train[dataset_train['evaluation'] == 2]
# data_1 = dataset_train[dataset_train['evaluation'] == 1]
# data_0 = dataset_train[dataset_train['evaluation'] == 0]

# def Data_aug(Dataset, num_aug, label_num):
    
#     data_aug = []
#     data_aug = pd.DataFrame(data_aug)
    
#     for dt in range(len(Dataset)):
#         Dataset_ = eda(Dataset['data_text'].reset_index(drop=True)[dt], num_aug = num_aug)
#         Dataset_ = pd.DataFrame(Dataset_)
#         Dataset_['data_text'] = Dataset_
#         Dataset_ = Dataset_.iloc[:, 1:]
#         Dataset_['evaluation'] = label_num
#         data_aug = pd.concat([Dataset_,data_aug], ignore_index=True)
        
#     return data_aug

# data_1 = Data_aug(data_1, 11, 1)
# data_0 = Data_aug(data_0, 9, 0)

# dataset_train_aug = pd.concat([data_2, data_1, data_0], ignore_index=True)
# dataset_train_aug = dataset_train_aug.sample(frac=1).reset_index(drop=True)

# train_data = CustomDataset(dataset_train_aug)
# test_data = CustomDataset(dataset_test)

#%%

'''normal'''

train_data = CustomDataset(dataset_train)
test_data = CustomDataset(dataset_test)

#%%

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)#, collate_fn=None)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)#, collate_fn=None)
total_steps = len(train_loader) * EPOCHS
warmup_step = int(total_steps * warmup_ratio)

pred_ls = []
label_ls = []

model = BertForSequenceClassification.from_pretrained(pretrain, num_labels=3,
                                                      # output_attentions=True,
                                                      # output_hidden_states = True,
                                                      # attention_probs_dropout_prob=0.5,
                                                      # hidden_dropout_prob=0.5,
                                                      ).to(device)
# model.resize_token_embeddings(len(tokenizer))
# assert model.config.output_attentions == True

# bert = model.bert
# dropout = model.dropout
# classifier = model.classifier
# model = nn.Sequential(bert, dropout, classifier)

# model=nn.DataParallel(model).to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss().to(device)
# weight=torch.tensor([0.4, 0.4, 0.2]) # 중립, 부정, 긍정
# loss_fnw = nn.CrossEntropyLoss(weight=weight)).to(device)

#%%

'''labelsmoothing'''

# class LabelSmoothingLoss(nn.Module):
#     """
#     With label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     """
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
#         assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         super(LabelSmoothingLoss, self).__init__()

#         smoothing_value = label_smoothing / (tgt_vocab_size - 2)
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value)
#         one_hot[self.ignore_index] = 0
#         self.register_buffer('one_hot', one_hot.unsqueeze(0))

#         self.confidence = 1.0 - label_smoothing

#     def forward(self, output, target):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         model_prob = self.one_hot.repeat(target.size(0), 1)
#         model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
#         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
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
    
loss_ls = LabelSmoothingLoss(smoothing = 0.3)

#%%

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
        
        
        # outputs = bert(input_ids = input_ids,
        #                token_type_ids = token_type_ids,
        #                attention_mask = attention_masks)
        
        # output = dropout(outputs[1])
        # logits = classifier(output)
        
        # loss = loss_fn(logits, labels)
        
        
        out = model(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                    labels=labels)
        logits = out[1]
        
        # loss = loss_fnw(logits, labels)
        
        loss = loss_ls(logits, labels)
        
        # loss = out[0]
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
          
        total_loss += loss.item()
          
        # _, predicted = torch.max(label_pred, 1)
        _, predicted = torch.max(logits, 1)
        correct_train += (predicted == labels).sum()
        total_train += len(labels)

    print("")
    print("epoch {} Train Loss {} train acc {}".format(i+1, total_loss/ total_train, correct_train / total_train))
    print("")
    
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
        
        pred = logits.argmax(dim=1).to(device)
        target = labels.view_as(pred).to(device)

        f1_mac = F1Score(average='macro', num_classes=3)
        f1_mac_ += f1_mac(pred.cpu(), target.cpu())
        recall_mac = Recall(average='macro', num_classes=3)
        recall_mac_ += recall_mac(pred.cpu(), target.cpu())
        precision_mac = Precision(average='macro', num_classes=3)
        precision_mac_ += precision_mac(pred.cpu(), target.cpu())

        f1_mic = F1Score(average='micro', num_classes=3)
        f1_mic_ += f1_mic(pred.cpu(), target.cpu())
        recall_mic = Recall(average='micro', num_classes=3)
        recall_mic_ += recall_mic(pred.cpu(), target.cpu())
        precision_mic = Precision(average='micro', num_classes=3)
        precision_mic_ += precision_mic(pred.cpu(), target.cpu())
        
        label_ls += target.tolist()
        pred_ls += pred.tolist()
        
    f1_macro = f1_score(label_ls, pred_ls, average='macro', zero_division=1)
    precision_macro = precision_score(label_ls, pred_ls, average='macro', zero_division=1)
    recall_macro = recall_score(label_ls, pred_ls, average='macro', zero_division=1)
    f1_micro = f1_score(label_ls, pred_ls, average='micro', zero_division=1)
    precision_micro = precision_score(label_ls, pred_ls, average='micro', zero_division=1)
    recall_micro = recall_score(label_ls, pred_ls, average='micro', zero_division=1)
                 
    f1_mac_ /= len(test_loader)
    recall_mac_ /= len(test_loader)
    precision_mac_ /= len(test_loader)
    f1_mic_ /= len(test_loader)
    recall_mic_ /= len(test_loader)
    precision_mic_ /= len(test_loader)
    print("")
    print("F1 macro :{} Recall macro :{} Precision macro :{}".format(f1_mac_.item(), recall_mac_.item(), precision_mac_.item()))
    print("F1 micro :{} Recall micro :{} Precision micro :{}".format(f1_mic_.item(), recall_mic_.item(), precision_mic_.item()))
    print("F1 macro :{} Recall macro :{} Precision macro :{}".format(f1_macro.item(), recall_macro.item(), precision_macro.item()))
    print("F1 micro :{} Recall micro :{} Precision micro :{}".format(f1_micro.item(), recall_micro.item(), precision_micro.item()))
    print("epoch :{} test acc :{}".format(i+1, correct_eval/total_eval))
    print("")
    print("#######end epoch:",i+1,"#######")
    print("")


#%%

def predict(predict_sentence):
    
    another_test = CustomDataset(predict_sentence)
    another_dataloader = DataLoader(another_test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    model.eval()
    
    for batch in tqdm(another_dataloader):
        batch = tuple(v.to(device) for v in batch)
        input_ids, token_type_ids, attention_masks, labels = batch
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        labels=labels
                        )
            logits = out[1]
            # outputs = bert(input_ids = input_ids,
            #                     token_type_ids = token_type_ids,
            #                     attention_mask = attention_masks)#,
            #                     #labels=labels)
            # pooler_output = outputs[1]
            # logits = classifier(pooler_output)
    
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

#%%
#test = '#마요패키지가 워낙 고급스럽게 잘 나왔고 글로우픽 1위 제품이라 믿고 샀는데컬러 질감 지속력 모두 가격대비 그냥 그런..?'
#test = '코스맥스 제조일때밖에 구매를 안해서 그때를 기준으로 얘기하겠습니다. 시빼테,대추차등등 이쁜색이 정말 많고 지금도 잘 쓰고있습니다. ..살짝 캐스터네츠이긴 한데 가격이 싸니까 그런점은 감안할 수 있습니다. 지속력도 좋은 편이구요.'
#test = '확실히 제품력은 비싼값을 하는 친구에요!! 저는 264호 100일 마른 장미를 샀어요~ 여름뮤트톤인 김고은씨가 화보에 바르고나와서 김고은 립스틱으로도 불린다고 하길래 덥석 집어왔더랬죠ㅋㅋㅋㅋ 264호는 쉬어타입인데 입술에 발랐을때 균일하게 미끄러지듯 발리면서 적당한 촉촉함에 각질부각도 안되고 무엇보다 주름끼임도 별로 없어서 좋았어요:) 발라보면 진짜 퀄리티 좋구나를 절절히 느낄수 있습니다!ㅋㅋㅋ 색은 본통으로 보면 많이 핑키쉬하고 약~~~간 어두운 말린장미색인데 막상 입술에 올리면 저한테는 살짝 웜하고 흰끼가 올라오는 것 같은 느낌이 들었어요..! 그렇다고 막 얼굴에서 뜨는것 같진 않은데 베스트까지는 아닌?? 저는 여름뮤트중에서도 조금 어두운 컬러가 베스트인 것 같은데 이컬러는 쿨톤도 쓸수는 있지만 젤착붙인 톤을 찾으라면 가을뮤트일것 같아요! 그중에서도 밝은쪽 뮤트인 소프트?? 어디까지나 저의 궁예입니다ㅋㅋㅋ 그리고 엄청 자잘한 은(?)펄이 있는데 발색했을 때 눈에 띄는 정도는 아니지만 조금 더 글로시 해보이는 느낌을 주는 것 같았어요!! 저는 매트 무펄 성애자로서 조금 아쉬웠던 점이었습니다ㅎㅎ 엄마는 350호를 갖고 계신데 진짜 예쁜 코랄색이에요 흰끼도 많이 안돌고 딱예쁜 화사한 코랄!! 저한테는 죽어라 안어울리는 컬러지만요..★ 근데 이건 면세에 밖에 없나봐요ㅠㅠ 이 립스틱 다 좋은데 단점을 꼽자면 냄새가 너무 구려요.....ㅎ...ㅎ 할머니루즈 냄새납니다@.@ 립스틱 말고 루즈 냄새요ㅋㅋㅋㅋㅋ 사실 외할머니가 쓰시는 립스틱도 랑콤이라서 그런걸 수도 있지만요..★  화장품에 향료 들어가는거 굉장히 싫어하는데도 이건 랑콤본사가서 무릎꿇고 향료좀 넣어달라고 하고 싶을 정도에요ㅋㅋㅋ 여튼 퀄리티 넘사벽으로 좋지만 색감이나 펄감이 좀 아쉬웠고 무엇보다 하루종일 코밑에 있는 입술에 바르고 다닐 냄새는 아니라서 굳굳줍니다!'
test = '좋긴 좋은데 가격대비..음..1위할 제품은 아닌듯. 커버력은 괜찮은데 지속력이.. 음..면세점에서 색상별로 샀는데 뭔가 속은기분'
#test = '최악'


another = pd.DataFrame({'data_text': [test],
                   'evaluation': [0]})

predict(another)

#%%

# end = 1
# while end == 1 :
#     test = input("")
#     if test == '0' :
#       break
#     else:
#       another = pd.DataFrame({'data_text': [test], 'evaluation': [0]})
#       predict(another)
#     print("\n")














