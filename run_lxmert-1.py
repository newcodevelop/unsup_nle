# import numpy as np
# import jsonlines
# from tqdm import tqdm
# import os

# import sys
# sys.path.append('../../exps/transformers/examples/research_projects/visual_bert')
# import os
# print('right file execution')

# from IPython.display import Image, display
# import PIL.Image
# import io
# import torch
# import numpy as np
# from processing_image import Preprocess
# from visualizing_image import SingleImageViz
# from modeling_frcnn import GeneralizedRCNN
# from utils import Config
# import utils
# from transformers import LxmertModel, LxmertTokenizer
# import pandas as pd
# import pickle

# vs_feat = torch.load('./vs_tensors_lxmert.pt')

# from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


# data = pd.read_csv('./MEMES_MY_DATASET_WITHOUT_OVERSAMPLING_new.csv')
# outliers = []
# for names in tqdm(list(data['Name'])):
#   if not os.path.exists('./my_meme_data/'+names):
#     outliers.append(names)

# data = data[~data['Name'].isin(outliers)]


# with open('train_translate.pkl', 'rb') as handle:
#     train = pickle.load(handle)

# with open('val_translate.pkl', 'rb') as handle:
#     validation = pickle.load(handle)

# with open('test_translate.pkl', 'rb') as handle:
#     test = pickle.load(handle)


# # all_translate = train+validation+test

# all_translate = {**train, **validation, **test}

# train_set = []

# for _,i in data.iterrows():

#     train_set.append({'text': all_translate[i['Name']], 

#                        'img': i['Name'],

#                        'label': i['Sarcasm']
#         })



# print(train_set)


# class CustomDataset(Dataset):

#     def __init__(self, ds, max_len=64):
#         self.ds = ds
#         self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
#         self.model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        
#         self.max_len = max_len
#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, index):

#         ds_idx = self.ds[index]
#         inputs = self.tokenizer(
#         ds_idx['text'],
#         padding="max_length",
#         max_length=self.max_len,
#         truncation=True,
#         return_token_type_ids=True,
#         return_attention_mask=True,
#         add_special_tokens=True,
#         return_tensors="pt"
#         )
#         # inputs = self.tokenizer(ds_idx['text'], padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt')


#         inputs['input_ids'] = inputs['input_ids'].squeeze(0)
#         inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
#         inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
#         # visual_embeds,visual_token_type_ids,visual_attention_mask = get_visual_embedding('./data/'+ds_idx['img'])
#         features,normalized_boxes = vs_feat[ds_idx['img']]

        
#         inputs.update({
#           "visual_feats": torch.squeeze(features),
#           "visual_pos": torch.squeeze(normalized_boxes)
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
#         self.vb =  LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

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
#         visual_feats = None,
#         visual_pos = None,
#         attention_mask = None,
#         visual_attention_mask = None,
#         token_type_ids = None,
#         inputs_embeds = None,
#         output_attentions= None,
#         output_hidden_states= None,
#         return_dict=None):
#         x = self.vb(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_feats=visual_feats, visual_pos=visual_pos, return_dict=False)
#         # print(x[-1])
#         logits = self.linear_relu_stack(x[-1])
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

# seed=42
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
        
#         print(loss)
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
#     print(roc_auc_score(act,pred_proba))










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
from transformers import VisualBertModel, BertTokenizer, VisualBertConfig, AutoConfig, VisualBertForQuestionAnswering
import pandas as pd
import pickle
import json

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler



# train_set  = []
# with jsonlines.open('../../../data/train.jsonl') as reader:
#   for obj in tqdm(reader):
#     train_set.append(obj)

# test_set  = []
# with jsonlines.open('../../../data/dev.jsonl') as reader:
#   for obj in tqdm(reader):
#     test_set.append(obj)

# data = pd.read_csv('./MEMES_MY_DATASET_WITHOUT_OVERSAMPLING_new.csv')
# outliers = []
# for names in tqdm(list(data['Name'])):
#   if not os.path.exists('./my_meme_data/'+names):
#     outliers.append(names)

# data = data[~data['Name'].isin(outliers)]


# with open('train_translate.pkl', 'rb') as handle:
#     train = pickle.load(handle)

# with open('val_translate.pkl', 'rb') as handle:
#     validation = pickle.load(handle)

# with open('test_translate.pkl', 'rb') as handle:
#     test = pickle.load(handle)


# # all_translate = train+validation+test

# all_translate = {**train, **validation, **test}

# train_set = []

# for _,i in data.iterrows():

#     train_set.append({'text': all_translate[i['Name']], 

#                        'img': i['Name'],

#                        'label': i['Sarcasm']
#         })



# print(train_set)






# script.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

print(f"Hi, {args.dataset}!")

if args.dataset.lower() not in ['actx', 'aokvqa', 'vqax']:
    raise Exception("Name should be between actx, aokvqa, vqa-x")


vs_feat = torch.load('./tensors/vs_tensors_train2014.pt')


base_image_dir = '/home/dibyanayan/unsup_nle' 




json_file_path_train = '/home/dibyanayan/unsup_nle/vqav2/v2_OpenEnded_mscoco_train2014_questions.json'



with open(json_file_path_train, 'r') as f:
    data_questions = json.load(f)


print(data_questions.keys())

questions = data_questions['questions']
print("Number of questions:", len(questions))



import re
from typing import Optional

filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

# source: https://github.com/allenai/allennlp-models/blob/a36aed540e605c4293c25f73d6674071ca9edfc3/allennlp_models/vision/dataset_readers/vqav2.py#L141
def id_from_filename(filename: str) -> Optional[int]:
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))


from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm

# root at which all images are stored
root = '/home/dibyanayan/unsup_nle/train2014'
file_names = [f for f in tqdm(listdir(root)) if isfile(join(root, f))]

print(questions[0])

filename_to_id = {root + "/" + file: id_from_filename(file) for file in file_names}
id_to_filename = {v:k for k,v in filename_to_id.items()}


cnt = 0
for i,j in zip(filename_to_id, id_to_filename):
    print(filename_to_id[i], id_to_filename[j])
    cnt+=1

    if cnt==4:
        break




import json

# Read annotations
f = open('/home/dibyanayan/unsup_nle/vqav2/v2_mscoco_train2014_annotations.json')

# Return JSON object as dictionary
data_annotations = json.load(f)
print(data_annotations.keys())

annotations = data_annotations['annotations']

print("Number of annotations:", len(annotations))



print(annotations[0])


from transformers import ViltConfig

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


from tqdm import tqdm

def get_score(count: int) -> float:
    return min(1.0, count / 3)

for annotation in tqdm(annotations):
    answers = annotation['answers']
    answer_count = {}
    for answer in answers:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1
    labels = []
    scores = []
    for answer in answer_count:
        if answer not in list(config.label2id.keys()):
            continue
        labels.append(config.label2id[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    annotation['labels'] = labels
    annotation['scores'] = scores

print(annotations[0])


labels = annotations[0]['labels']
print([config.id2label[label] for label in labels])

scores = annotations[0]['scores']
print(scores)


import torch
from PIL import Image

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations, tokenizer):
        self.questions = questions
        self.annotations = annotations
        # self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = 128

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        # image = Image.open(id_to_filename[annotation['image_id']])
        image_path = id_to_filename[annotation['image_id']]
        text = questions['question']


        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt')


        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
        # visual_embeds,visual_token_type_ids,visual_attention_mask = get_visual_embedding('./data/'+ds_idx['img'])
        visual_embeds, normalized_boxes = vs_feat[image_path]

        # visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        # visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        
        # inputs.update({
        #   "visual_embeds": torch.squeeze(visual_embeds),
        #   "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
        #   "visual_attention_mask": torch.squeeze(visual_attention_mask)
        # })

        inputs.update({
            "visual_feats": torch.squeeze(visual_embeds),
            "visual_pos": torch.squeeze(normalized_boxes)
            })
        # encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in inputs.items():
          inputs[k] = v.squeeze()
        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))
        for label, score in zip(labels, scores):
              targets[label] = score
        inputs["labels"] = targets

        return inputs

# from transformers import ViltProcessor

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

from transformers import AutoTokenizer, VisualBertForQuestionAnswering, LxmertModel
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

dataset = VQADataset(questions=questions,
                     annotations=annotations,
                     tokenizer=tokenizer)

for i in dataset:
    print(i)
    break


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

def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  visual_feats = [item['visual_feats'] for item in batch]
  visual_pos = [item['visual_pos'] for item in batch]
  labels = [item['labels'] for item in batch]

  

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['visual_feats'] = torch.stack(visual_feats)
  batch['visual_pos'] = torch.stack(visual_pos)
  batch['labels'] = torch.stack(labels)

  return batch

train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=256, shuffle=True)


# for i in train_dataloader:
#     print(i)
#     print(tokenizer.batch_decode(i['input_ids']))
#     for j in i['labels'].argmax(dim=-1).detach().cpu().numpy():
#         print(config.id2label[j])

#     break


from torch import nn


class LxmertForQuestionAnswering(nn.Module):
    def __init__(self):
        super(LxmertForQuestionAnswering, self).__init__()
        self.lxmert =  LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        self.cls = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(768, 3129)
        )

        self.dropout = nn.Dropout(0.1)
        self.num_labels = 3129

    def forward(self,
        input_ids = None,
        visual_feats = None,
        visual_pos = None,
        attention_mask = None,
        visual_attention_mask = None,
        token_type_ids = None,
        inputs_embeds = None,
        output_attentions= None,
        output_hidden_states= None,
        labels=None,
        return_dict=None):

        input_ids = batch['input_ids']
        visual_feats = batch['visual_feats']
        visual_pos = batch['visual_pos']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        labels = batch['labels']
        
        # Get the index of the last text token
        index_to_gather = attention_mask.sum(1) - 2  # as in original code

        outputs = self.lxmert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_feats=visual_feats, visual_pos=visual_pos, return_dict=False)
        


        sequence_output = outputs[0]

        # TO-CHECK: From the original code
        index_to_gather = (
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        )
        pooled_output = torch.gather(sequence_output, 1, index_to_gather)

        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        # loss = None
        
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        log_softmax = nn.LogSoftmax(dim=-1)
        reshaped_logits = log_softmax(reshaped_logits)
        loss = loss_fct(reshaped_logits, labels.contiguous())

        return loss
       



model = LxmertForQuestionAnswering()
device = 'cuda'
model.to(device)

print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(4):  # loop over the dataset multiple times
    avg_l = []
    for xcount, batch in tqdm(enumerate(train_dataloader)):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = model(**batch)
        avg_l.append(loss.item())
        if xcount%10==0:
            print("Loss:", loss.item(), xcount)
        loss.backward()
        optimizer.step()
    print("Epoch {} : Average Loss {}".format(epoch+1, np.mean(np.asarray(avg_l))))



torch.save(model.state_dict(), './models/lxmert_vqav2.pt')

print('LxMERT Training Completed and Model saved')