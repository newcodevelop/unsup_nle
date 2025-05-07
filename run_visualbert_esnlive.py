import numpy as np
import jsonlines
from tqdm import tqdm
import os

import sys
# sys.path.append('../../exps/transformers/examples/research_projects/visual_bert')
import os
print('right file execution')

import PIL.Image
import io
import torch
import numpy as np
# from processing_image import Preprocess
# from visualizing_image import SingleImageViz
# from modeling_frcnn import GeneralizedRCNN
# from utils import Config
# import utils
from transformers import AutoTokenizer
from transformers import VisualBertModel, BertTokenizer, VisualBertConfig, AutoConfig, VisualBertForQuestionAnswering
import pandas as pd
import pickle
import json

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)   # expect "esnlive"
args = parser.parse_args()
if args.dataset.lower() != "esnlive":
    raise Exception("Use  --dataset esnlive")


# vs_feat = torch.load('./tensors/vs_tensors_train2014.pt')


# base_image_dir = '/home/dibyanayan/unsup_nle' 




# json_file_path_train = '/home/dibyanayan/unsup_nle/vqav2/v2_OpenEnded_mscoco_train2014_questions.json'

feat_path = "./tensors/vs_tensors_esnlive.pt"        
csv_path  = "/home/dibyanayan/unsup_nle/e-ViL/data/esnlive_train.csv"
img_root  = "/home/dibyanayan/unsup_nle/flickr30k_images/flickr30k_images/flickr30k_images"
save_dir  = "./models"


# with open(json_file_path_train, 'r') as f:
#     data_questions = json.load(f)


# print(data_questions.keys())

# questions = data_questions['questions']
# print("Number of questions:", len(questions))

vs_feat = torch.load(feat_path)
vs_feat = {os.path.basename(k): v for k, v in vs_feat.items()}
df = pd.read_csv(csv_path)
avail = set(vs_feat.keys())
df = df[df["Flickr30kID"].isin(avail)]
print(f"Prepared {len(df):,} e‑SNLI‑VE samples")
label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
questions, annotations = [], []
for idx, row in df.iterrows():
    questions.append({"question_id": idx, "question": row["hypothesis"]})
    annotations.append(
        {
            "image_id": int(row["Flickr30kID"].split(".")[0]),
            "labels": [label2id[row["gold_label"]]],
            "scores": [1.0],
        }
    )

id_to_filename = {
    int(name.split(".")[0]): os.path.join(img_root, name) for name in avail
}

# import re
# from typing import Optional

# filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

# # source: https://github.com/allenai/allennlp-models/blob/a36aed540e605c4293c25f73d6674071ca9edfc3/allennlp_models/vision/dataset_readers/vqav2.py#L141
# def id_from_filename(filename: str) -> Optional[int]:
#     match = filename_re.fullmatch(filename)
#     if match is None:
#         return None
#     return int(match.group(1))


# from os import listdir
# from os.path import isfile, join
# from tqdm.auto import tqdm

# # root at which all images are stored
# root = '/home/dibyanayan/unsup_nle/train2014'
# file_names = [f for f in tqdm(listdir(root)) if isfile(join(root, f))]

# print(questions[0])

# filename_to_id = {root + "/" + file: id_from_filename(file) for file in file_names}
# id_to_filename = {v:k for k,v in filename_to_id.items()}


# cnt = 0
# for i,j in zip(filename_to_id, id_to_filename):
#     print(filename_to_id[i], id_to_filename[j])
#     cnt+=1

#     if cnt==4:
#         break




# import json

# # Read annotations
# f = open('/home/dibyanayan/unsup_nle/vqav2/v2_mscoco_train2014_annotations.json')

# # Return JSON object as dictionary
# data_annotations = json.load(f)
# print(data_annotations.keys())

# annotations = data_annotations['annotations']

# print("Number of annotations:", len(annotations))



# print(annotations[0])


# from transformers import ViltConfig

# config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


# from tqdm import tqdm

# def get_score(count: int) -> float:
#     return min(1.0, count / 3)

# for annotation in tqdm(annotations):
#     answers = annotation['answers']
#     answer_count = {}
#     for answer in answers:
#         answer_ = answer["answer"]
#         answer_count[answer_] = answer_count.get(answer_, 0) + 1
#     labels = []
#     scores = []
#     for answer in answer_count:
#         if answer not in list(config.label2id.keys()):
#             continue
#         labels.append(config.label2id[answer])
#         score = get_score(answer_count[answer])
#         scores.append(score)
#     annotation['labels'] = labels
#     annotation['scores'] = scores

# print(annotations[0])


# labels = annotations[0]['labels']
# print([config.id2label[label] for label in labels])

# scores = annotations[0]['scores']
# print(scores)


import torch
from PIL import Image

# class VQADataset(torch.utils.data.Dataset):
#     """VQA (v2) dataset."""

#     def __init__(self, questions, annotations, tokenizer):
#         self.questions = questions
#         self.annotations = annotations
#         # self.processor = processor
#         self.tokenizer = tokenizer
#         self.max_len = 128

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         # get image + text
#         annotation = self.annotations[idx]
#         questions = self.questions[idx]
#         # image = Image.open(id_to_filename[annotation['image_id']])
#         image_path = id_to_filename[annotation['image_id']]
#         text = questions['question']


#         inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt')


#         inputs['input_ids'] = inputs['input_ids'].squeeze(0)
#         inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
#         inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
#         # visual_embeds,visual_token_type_ids,visual_attention_mask = get_visual_embedding('./data/'+ds_idx['img'])
#         visual_embeds, _ = vs_feat[image_path]

#         visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
#         visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        
#         inputs.update({
#           "visual_embeds": torch.squeeze(visual_embeds),
#           "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
#           "visual_attention_mask": torch.squeeze(visual_attention_mask)
#         })

#         # encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
#         # remove batch dimension
#         for k,v in inputs.items():
#           inputs[k] = v.squeeze()
#         # add labels
#         labels = annotation['labels']
#         scores = annotation['scores']
#         # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
#         targets = torch.zeros(len(config.id2label))
#         for label, score in zip(labels, scores):
#               targets[label] = score
#         inputs["labels"] = targets

#         return inputs

class ESNLIVE_VB_Dataset(Dataset):
    def __init__(self, questions, annotations, tokenizer, max_len=128):
        self.questions = questions
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        text  = self.questions[idx]["question"]
        img_key = os.path.basename(id_to_filename[annot["image_id"]])

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        visual_embeds, _ = vs_feat[img_key]
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update(
            {
                "visual_embeds": visual_embeds.squeeze(),
                "visual_token_type_ids": visual_token_type_ids.squeeze(),
                "visual_attention_mask": visual_attention_mask.squeeze(),
            }
        )

        targets = torch.zeros(3)
        for l, s in zip(annot["labels"], annot["scores"]):
            targets[l] = s
        inputs["labels"] = targets
        return inputs


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([item[k] for item in batch])
    return out

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
dataset   = ESNLIVE_VB_Dataset(questions, annotations, tokenizer)
train_dl  = DataLoader(dataset, collate_fn=collate_fn, batch_size=64, shuffle=True)
vb_cfg = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa", num_labels=3)
vb_cfg.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
vb_cfg.label2id = {v: k for k, v in vb_cfg.id2label.items()}

# from transformers import ViltProcessor

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# from transformers import AutoTokenizer, VisualBertForQuestionAnswering, LxmertForQuestionAnswering
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# dataset = VQADataset(questions=questions,
#                      annotations=annotations,
#                      tokenizer=tokenizer)

# for i in dataset:
#     print(i)
#     break


# cnt = 0
# for i in dataset[17]['labels']:
#     if i.item()==1:
#         print(cnt)
#         cnt +=1

# labels = torch.nonzero(dataset[17]['labels']).squeeze().tolist()

# print(labels)

# labels = [labels]

# print([config.id2label[label] for label in labels])

# print(cnt)


# print(dataset[17]['labels'].shape)


# model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa",id2label=config.id2label,
#                                                  label2id=config.label2id)

# model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

# model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased",id2label=config.id2label,
#                                                 label2id=config.label2id)

# print(model)




from torch.utils.data import DataLoader

# def collate_fn(batch):
#   input_ids = [item['input_ids'] for item in batch]
#   visual_embeds = [item['visual_embeds'] for item in batch]
#   attention_mask = [item['attention_mask'] for item in batch]
#   token_type_ids = [item['token_type_ids'] for item in batch]
#   visual_attention_mask = [item['visual_attention_mask'] for item in batch]
#   visual_token_type_ids = [item['visual_token_type_ids'] for item in batch]
#   labels = [item['labels'] for item in batch]
model = VisualBertForQuestionAnswering.from_pretrained(
    "uclanlp/visualbert-vqa",
    config=vb_cfg,
    ignore_mismatched_sizes=True,  
)
  
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(model)

  # create new batch
#   batch = {}
#   batch['input_ids'] = torch.stack(input_ids)
#   batch['attention_mask'] = torch.stack(attention_mask)
#   batch['token_type_ids'] = torch.stack(token_type_ids)
#   batch['visual_embeds'] = torch.stack(visual_embeds)
#   batch['visual_token_type_ids'] = torch.stack(visual_token_type_ids)
#   batch['visual_attention_mask'] = torch.stack(visual_attention_mask)
#   batch['labels'] = torch.stack(labels)

#   return batch

# train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=256, shuffle=True)


# for i in train_dataloader:
#     print(i)
#     print(tokenizer.batch_decode(i['input_ids']))
#     for j in i['labels'].argmax(dim=-1).detach().cpu().numpy():
#         print(config.id2label[j])

#     break

# device = 'cuda'
# model.to(device)


optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()



for epoch in range(4):
    epoch_loss = []
    for step, batch in enumerate(tqdm(train_dl)):
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step()
        epoch_loss.append(loss.item())
        if step % 10 == 0:
            print(f"Epoch {epoch+1}  step {step}  loss {loss.item():.4f}")
    print(f"Epoch {epoch+1}  average loss {np.mean(epoch_loss):.4f}")



torch.save(model.state_dict(), os.path.join(save_dir, "visualbert_esnlive.pt"))

print('VisualBERT Training Completed and Model saved')


# # code ends here/......

# print(0/0)





# json_file_path_train = '/home/dibyanayan/unsup_nle/data/explanation_dataset.json'  

# json_file_path_test = '/home/dibyanayan/unsup_nle/data/explanation_dataset_test.json'  

# with open(json_file_path_train, 'r') as f:
#     train_data = json.load(f)

# with open(json_file_path_test, 'r') as f:
#     test_data = json.load(f)

# train_set_lab, test_set_lab = [],[]
# for entry in train_data:
#     if entry['dataset'].lower()==args.dataset:
    
#         full_path = os.path.join(base_image_dir, entry['img_path'])
#         if os.path.exists(full_path):
#             train_set_lab.append(entry['answer'])

# for cnt, entry in enumerate(test_data):
#     # print(entry)
#     if test_data[str(entry)]['dataset'].lower()==args.dataset:
#         full_path = os.path.join(base_image_dir, test_data[str(entry)]['img_path'])
#         if os.path.exists(full_path):
#             if "" not in test_data[str(entry)]['answer']:
#                 test_set_lab.extend(test_data[str(entry)]['answer'])


# # print(train_set_lab)

# train_set_lab = list(set(train_set_lab))
# test_set_lab = list(set(test_set_lab))


# marker = 0

# kk = []

# for i in test_set_lab:
#     if i not in train_set_lab:
#         kk.append(i)
#         marker+=1
       

# print(marker)

# print(len(kk))
# print(len(train_set_lab), len(test_set_lab))



# # from transformers import ViltConfig

# # config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


# # predefined_labs = config.label2id.keys()


# # marker = []
# # for i in train_set_lab + test_set_lab:
# #     if i not in predefined_labs:
# #         marker.append(i)

# # print(marker, len(marker))




# print(0/0)












# print(0/0)




# json_file_path_train = '/home/dibyanayan/unsup_nle/data/explanation_dataset.json'  

# json_file_path_test = '/home/dibyanayan/unsup_nle/data/explanation_dataset_test.json'  

# with open(json_file_path_train, 'r') as f:
#     data = json.load(f)


# train_set = []

# for entry in data:
#     if entry['dataset'].lower()==args.dataset:
#         full_path = os.path.join(base_image_dir, entry['img_path'])
#         if os.path.exists(full_path):
#             train_set.append({'text':  entry['question'], 'img': entry['img_path'], 'label':  entry['answer']})
            

# print(train_set[:10])


# labs = list(set([i['label'] for i in train_set]))









# print(labs[:5], len(labs))


# print(0/0)




# class CustomDataset(Dataset):

#     def __init__(self, ds, max_len=64):
#         self.ds = ds
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         self.model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        
#         self.max_len = max_len
#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, index):

#         ds_idx = self.ds[index]
#         inputs = self.tokenizer(ds_idx['text'], padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt')


#         inputs['input_ids'] = inputs['input_ids'].squeeze(0)
#         inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
#         inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
#         # visual_embeds,visual_token_type_ids,visual_attention_mask = get_visual_embedding('./data/'+ds_idx['img'])
#         visual_embeds,visual_token_type_ids,visual_attention_mask = vs_feat[ds_idx['img']]

        
#         inputs.update({
#           "visual_embeds": torch.squeeze(visual_embeds),
#           "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
#           "visual_attention_mask": torch.squeeze(visual_attention_mask)
#         })

#         return inputs, int(ds_idx['label'])

# t = CustomDataset(train_set)
# # te = CustomDataset(test_set)
# # train_dev_sets = torch.utils.data.ConcatDataset([t, te])
# # torch.manual_seed(42)
# # lt = len(train_dev_sets)
# # print(lt)
# # t, te = torch.utils.data.random_split(train_dev_sets, [lt-500, 500])

# torch.manual_seed(123)
# t_p,te_p = torch.utils.data.random_split(t,[5908,1478])


# torch.manual_seed(123)
# t_p,v_p = torch.utils.data.random_split(t_p,[5022,886])


# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(t_p, shuffle=True, batch_size=32)
# eval_dataloader = DataLoader(te_p, shuffle=True, batch_size=1)



# import torch
# from torch import nn
# class VB(nn.Module):
#     def __init__(self):
#         super(VB, self).__init__()
#         self.vb = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
#         self.linear_relu_stack = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(768, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3),
#         )

#     def forward(self,
#         input_ids = None,
#         attention_mask = None,
#         token_type_ids = None,
#         position_ids = None,
#         head_mask = None,
#         inputs_embeds = None,
#         visual_embeds = None,
#         visual_attention_mask = None,
#         visual_token_type_ids = None,
#         image_text_alignment = None,
#         output_attentions = None,
#         output_hidden_states = None,
#         return_dict = None):
#         x = self.vb(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
#         logits = self.linear_relu_stack(x.pooler_output)
#         return logits

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

# vb = VB()

# vb = nn.DataParallel(vb)
# vb.to(device)


# from torch.optim import AdamW

# optimizer = AdamW(vb.parameters(), lr=5e-5)

# from transformers import get_scheduler

# num_epochs = 25
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )


# import random

# import numpy as np

# seed=123
# random.seed(seed)     # python random generator
# np.random.seed(seed)  # numpy random generator

# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# from tqdm.auto import tqdm
# loss_fn = nn.CrossEntropyLoss()
# progress_bar = tqdm(range(num_training_steps))
# device = 'cuda'


# from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
# from scipy.special import softmax
# for epoch in range(num_epochs):
#     vb.train()
#     for batch in train_dataloader:
       
#         batch_inp = {k: v.to(device) for k, v in batch[0].items()}

#         # print(batch)
#         outputs = vb(**batch_inp)
#         # print(outputs)
#         loss = loss_fn(outputs,batch[1].to(device))
        
#         # print(loss)
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)
#     vb.eval()
#     print('epoch {}'.format(epoch))
#     pred = []
#     act = []
#     pred_proba = []
#     for batch in eval_dataloader:
#       with torch.no_grad():
#         batch_inp = {k: v.to(device) for k, v in batch[0].items()}
#         outputs = vb(**batch_inp)
#         # print(outputs.detach().cpu().numpy())
#         act.append(batch[1].detach().cpu().numpy()[0])

#         pred.append(np.argmax(outputs.detach().cpu().numpy()[0]))
#         pred_proba.append(np.max(softmax(outputs.detach().cpu().numpy()[0])))


    
#     print(f1_score(act,pred,average='macro'))
#     print(f1_score(act,pred,average='weighted'))
#     print(accuracy_score(act,pred))
#     #print(roc_auc_score(act,pred_proba))


