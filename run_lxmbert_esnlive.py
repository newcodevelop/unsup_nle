
import numpy as np
# import jsonlines
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

from transformers import VisualBertModel, BertTokenizer, VisualBertConfig, AutoConfig, VisualBertForQuestionAnswering
from transformers import AutoTokenizer, LxmertModel
import pandas as pd
import pickle
import json
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

# print(f"Hi, {args.dataset}!")

if args.dataset.lower() != 'esnlive':
   raise Exception("Use --dataset esnlive for this script")

vs_feat = torch.load('./tensors/vs_tensors_esnlive.pt')
vs_feat = {os.path.basename(k): v for k, v in vs_feat.items()}


csv_path = "/home/dibyanayan/unsup_nle/e-ViL/data/esnlive_train.csv"
img_root = "/home/dibyanayan/unsup_nle/flickr30k_images/flickr30k_images/flickr30k_images"

df = pd.read_csv(csv_path)
available = set(vs_feat.keys())                        
df = df[df['Flickr30kID'].isin(available)]
label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}

questions, annotations = [], []
for idx, row in df.iterrows():
    questions.append(
        {"question_id": idx, "question": row["hypothesis"]}
    )
    annotations.append(
        {
            "image_id": int(row["Flickr30kID"].split(".")[0]),
            "labels": [label2id[row["gold_label"]]],
            "scores": [1.0],
        }
    )

id_to_filename = {}
for key in vs_feat.keys():                      
                                               
    fname = os.path.basename(key)              
    img_id = int(os.path.splitext(fname)[0])    
    id_to_filename[img_id] = os.path.join(img_root, fname)

print(f"Prepared {len(questions)} e‑SNLI‑VE samples")

class ESNLIVE_Dataset(Dataset):
    def __init__(self, questions, annotations, tokenizer, max_len=128):
        self.questions = questions
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        ques  = self.questions[idx]

        text = ques["question"]
        image_path = id_to_filename[annot["image_id"]]
        vis_key = os.path.basename(image_path)         

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        visual_embeds, visual_pos = vs_feat[vis_key]
        inputs.update(
            {
                "visual_feats": visual_embeds.squeeze(),
                "visual_pos": visual_pos.squeeze(),
            }
        )

        targets = torch.zeros(3)
        for label, score in zip(annot["labels"], annot["scores"]):
            targets[label] = score
        inputs["labels"] = targets

        return inputs


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([item[k] for item in batch])
    return out


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
dataset   = ESNLIVE_Dataset(questions, annotations, tokenizer)
train_dl  = DataLoader(dataset, collate_fn=collate_fn, batch_size=64, shuffle=True)

class LxmertForNLI(nn.Module):
    def __init__(self):
        super().__init__()
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(768, 3)         
        self.num_labels = 3
        self.loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.log_sm  = nn.LogSoftmax(dim=-1)

    def forward(self, **batch):
        outputs = self.lxmert(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            visual_feats=batch["visual_feats"],
            visual_pos=batch["visual_pos"],
            return_dict=False,
        )

        seq_out = outputs[0]                              
        gather_idx = batch["attention_mask"].sum(1) - 2    
        gather_idx = gather_idx.unsqueeze(-1).unsqueeze(-1).expand(
            gather_idx.size(0), 1, seq_out.size(-1)
        )
        pooled = torch.gather(seq_out, 1, gather_idx).squeeze(1) 

        logits = self.cls(self.dropout(pooled))           
        loss = self.loss_fct(self.log_sm(logits), batch["labels"])
        return loss


device = "cuda" if torch.cuda.is_available() else "cpu"
model = LxmertForNLI().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(4):
    epoch_loss = []
    for step, batch in enumerate(tqdm(train_dl)):
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        loss = model(**batch)
        loss.backward()
        optim.step()
        epoch_loss.append(loss.item())
        if step % 10 == 0:
            print(f"Epoch {epoch+1}  step {step}  loss {loss.item():.4f}")
    print(f"Epoch {epoch+1}  average loss {np.mean(epoch_loss):.4f}")

os.makedirs("./models", exist_ok=True)
torch.save(model.state_dict(), "./models/lxmert_esnlive.pt")
print("Training complete – model saved to  ./models/lxmert_esnlive.pt")