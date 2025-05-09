
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import torch
from torch import nn
from transformers import ViltConfig
from tqdm import tqdm

import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


import sys
sys.path.append('/home/dibyanayan/unsup_nle/transformers-research-projects/visual_bert')


import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, ViltConfig, ViltForQuestionAnswering, VisualBertForQuestionAnswering, LxmertModel, AutoTokenizer

device = 'cuda'

from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

device = 'cuda'
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.MODEL.device = device
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

frcnn.eval()

def get_visual_embedding(img_paths):
    images, sizes, scales_yx = image_preprocess(img_paths) # img_paths -> list of image paths
    output_dict = frcnn(
      images,
      sizes,
      scales_yx=scales_yx,
      padding="max_detections",
      max_detections=frcnn_cfg.max_detections,
      return_tensors="pt",
    )
    features = output_dict.get("roi_features")
    normalized_boxes = output_dict.get("normalized_boxes")
    return features, normalized_boxes


print(config.label2id)

num_labels = 3129


import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

print(f"Model: {args.model}")

#Run checks






if args.model.lower() not in ['visualbert', 'lxmert']:
    raise Exception("Model should be between visualbert, lxmert")





# output_dir = "src/generated_images_lxmert_ours"
# os.makedirs(output_dir, exist_ok=True)



if args.model.lower()=='lxmert':
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
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

            input_ids = inputs['input_ids']
            visual_feats = inputs['visual_feats']
            visual_pos = inputs['visual_pos']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']

            # labels = batch['labels']
            
            # Get the index of the last text token
            index_to_gather = attention_mask.sum(1) - 2  # as in original code

            outputs = self.lxmert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, visual_feats=visual_feats, visual_pos=visual_pos, return_dict=False)
            


            sequence_output = outputs[0]

            # TO-CHECK: From the original code
            index_to_gather = (
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
            )
            pooled_output_ = torch.gather(sequence_output, 1, index_to_gather)

            pooled_output = self.dropout(pooled_output_)
            logits = self.cls(pooled_output).argmax(dim=-1)

            return logits, pooled_output_
    
    model = LxmertForQuestionAnswering()
    model.load_state_dict(torch.load('../models/lxmert_vqav2.pt', weights_only=True))
    model.eval()
    model.to(device)

if args.model.lower()=='visualbert':
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
    model.load_state_dict(torch.load('../models/vb_vqav2.pt', weights_only=True))
    model.eval()
    model.to(device)











from datasets import load_dataset

# Stream the dataset instead of downloading it completely
dataset = load_dataset("Graphcore/vqa", split="validation", streaming=True, trust_remote_code=True)

# Take only the first N examples
dataset = dataset.take(4096)



BATCH_SIZE = 16  # Change as needed

# def collate_fn(batch):
#     texts = [i['question'] for i in batch]
#     image_paths = [
#         os.path.join("../val2014/", i['image_id'].split('::')[0].split('://')[-1].split('/')[-1])
#         for i in batch
#     ]
   

#     return {"image_paths": image_paths, "texts": texts}



def collate_fn(batch):
    texts = [i['question'] for i in batch]
    image_paths = [
        os.path.join("../val2014/", i['image_id'].split('::')[0].split('://')[-1].split('/')[-1])
        for i in batch
    ]
    images = [Image.open(path).convert("RGB") for path in image_paths]

    visual_feats = []
    visual_boxes = []

    for path in image_paths:
        feats, boxes = get_visual_embedding(path)
        visual_feats.append(feats)
        visual_boxes.append(boxes)

    # Stack visual features and boxes
    visual_feats = torch.stack(visual_feats).to('cuda')  # shape: [B, num_boxes, feat_dim]
    visual_boxes = torch.stack(visual_boxes).to('cuda')  # shape: [B, num_boxes, 4]

    # Tokenize text with padding
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    encoding = {k: v.to('cuda') for k, v in encoding.items()}

    return {
        **encoding,
        "visual_feats": visual_feats.squeeze(),
        "visual_pos": visual_boxes.squeeze()
       
        
    }, { "image_paths": image_paths, "texts": texts,
        "images": images}

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

import torch
from transformers import BertTokenizer, BertModel
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to('cuda')

# Set model to evaluation mode
bert_model.eval()
    

def extract_bert_embeddings(texts):
    """
    Extract BERT embeddings for the input text.
    
    Args:
        text (str): Input text to generate embeddings for
        model_name (str): Name of the BERT model to use
        
    Returns:
        torch.Tensor: Sentence embedding (averaged token embeddings)
    """
    # Load pre-trained model and tokenizer
    
    # Tokenize the input text
    encoding = bert_tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt',
    ).to('cuda')
    
    # Get input tensors
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Generate embeddings
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        
    # Get the embeddings from the last hidden state
    last_hidden_states = outputs.last_hidden_state
    
    # Average the token embeddings to get sentence embeddings
    # Using attention mask to ignore padding tokens
    sentence_embeddings = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
    
    return sentence_embeddings



# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

k = {}

# for pp, batch in enumerate(dataloader):
#     # print(batch)
#     inputs = batch[0]
#     # for i in inputs:
#     #     print(inputs[i].shape)
#     with torch.no_grad():
#         pred_lab, pooled_output = model(**inputs)
#     # print(batch[1])
#     # print(pred_lab, pooled_output.shape)
#     pooled_output = pooled_output.squeeze()

#     print(pooled_output.shape)
#     print(0/0)



for pp, batch in enumerate(dataloader):
    # print(batch)
    print(batch)

    break


import json
import re

# Load the JSON file
# with open('/home/dibyanayan/unsup_nle/src/k_normal_0_8_lxmert_vqa.json', 'r') as file:
#     data = json.load(file)


with open('/home/dibyanayan/unsup_nle/src/k_ours_lxmert_vqa.json', 'r') as file:
    data = json.load(file)


# Create a list to store the extracted information
result = []
multimodal_repr = []
concept_repr = []
# Iterate through each entry in the JSON
cnt = 0
inner_count = 0
prompts = []
for section, batch in tqdm(zip(data.values(), dataloader)):
    inputs = batch[0]
    # for i in inputs:
    #     print(inputs[i].shape)
    with torch.no_grad():
        pred_lab, pooled_output = model(**inputs)
    # print(batch[1])
    # print(pred_lab, pooled_output.shape)
    pooled_output = pooled_output.squeeze()
    multimodal_repr.append(pooled_output)
    img_paths = batch[1]['image_paths']
    texts = batch[1]['texts']
    

    expls = []


    for entry, txt, ip in zip(section.values(), texts, img_paths):
        # Extract question using regex
        question_match = re.search(r'Question: (.*?) Answer: (.*?)\.', entry)
        if question_match:
            question = question_match.group(1)
            answer = question_match.group(2)
            
            # Extract explanation (text after "because")
            explanation_match = re.search(r'because (.*?)(?:\n|$)', entry)
            explanation = explanation_match.group(1) if explanation_match else ""

            expls.append(explanation)
            prompts.append(explanation)
            
            
            if question==txt:
                # Add to result list
                result.append({
                    "question": question,
                    "answer": config.label2id[answer],
                    "explanation": explanation,
                    "text": txt,
                    "image_path": ip
                })
            else:
                continue
        
    concept_repr.append(extract_bert_embeddings(expls))

    
    
    cnt+=1

    if cnt==8:
        break


    

multimodal_repr = torch.stack(multimodal_repr).squeeze()
concept_repr = torch.stack(concept_repr).squeeze()



multimodal_repr = multimodal_repr.reshape(multimodal_repr.shape[0]*multimodal_repr.shape[1], multimodal_repr.shape[-1])
concept_repr = concept_repr.reshape(concept_repr.shape[0]*concept_repr.shape[1], concept_repr.shape[-1])
multimodal_repr = multimodal_repr.cpu()
concept_repr = concept_repr.cpu()

print(multimodal_repr.shape, concept_repr.shape)



import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")



output_dir = "./generated_images_lxmert_vqa_ours"
os.makedirs(output_dir, exist_ok=True)

# Process prompts in batches
batch_size = 64  # Adjust based on your GPU memory
num_batches = (len(prompts) + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(prompts))
    batch_prompts = prompts[start_idx:end_idx]
    
    print(f"Generating batch {batch_idx+1}/{num_batches} with {len(batch_prompts)} prompts")
    
    # Generate images for the entire batch at once
    results = pipe(prompt=batch_prompts, num_inference_steps=50, output_type="pil").images
    
    # Save each image in the batch
    for i, image in enumerate(results):
        prompt_idx = start_idx + i
        image.save(os.path.join(output_dir, f"image_{prompt_idx+1:02d}.png"))
        print(f"Saved image {prompt_idx+1}/16: {prompts[prompt_idx]}")

print(f"All images saved in '{output_dir}'")










print(0/0)
# Print the result (or save to a new file)
# print(json.dumps(result, indent=4))


import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
import random
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score



# Load CLIP model and processor
# model_id = "openai/clip-vit-base-patch32"
# model = CLIPModel.from_pretrained(model_id).to('cuda')
# processor = CLIPProcessor.from_pretrained(model_id)
# tokenizer = CLIPTokenizer.from_pretrained(model_id)

# # Function to get text embedding
# def get_text_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to('cuda')
#     with torch.no_grad():
#         text_features = model.get_text_features(**inputs)
#     return text_features.cpu()

# # Function to get image embedding
# def get_image_embedding(image_path):
#     image = Image.open(image_path)
#     inputs = processor(images=image, text=None, return_tensors="pt").to('cuda')
#     with torch.no_grad():
#         image_features = model.get_image_features(inputs["pixel_values"])
#     return image_features.cpu()

# # Process the data
# processed_data = []
# # multimodal_repr = []
# concept_repr = []
# y_p = []

y_p  = [item["answer"] for item in result]

# from tqdm import tqdm

# for item in tqdm(result):
#     # Get question embedding
#     # question_embedding = get_text_embedding(item["question"])
    
#     # Get image embedding
#     # image_embedding = get_image_embedding(item["image_path"])
    
#     # Create multimodal embedding (question + image)
#     # multimodal_embedding = question_embedding + image_embedding
    
#     # Get explanation embedding
#     explanation_embedding = get_text_embedding(item["explanation"])

#     # multimodal_repr.append(multimodal_embedding)
#     concept_repr.append(explanation_embedding)
#     y_p.append(item["answer"])
    
#     # # Add embeddings to the item
#     # processed_item = item.copy()
#     # processed_item["multimodal_embedding"] = multimodal_embedding
#     # processed_item["explanation_embedding"] = explanation_embedding

    
#     # processed_data.append(processed_item)

# multimodal_repr = multimodal_repr.cpu()
# concept_repr = torch.stack(concept_repr).squeeze()



"""

import torch
import torch.nn as nn

class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.hidden = nn.Linear(input_size, 128)
        self.act = nn.ReLU()
        self.output = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)  # No activation needed here
        return x

# Loss function for multiclass classification
criterion = nn.CrossEntropyLoss()



def get_sim(multimodal_repr, concept_repr1, y_p):

    X = np.concatenate([multimodal_repr.numpy(),concept_repr1.numpy()],axis=-1) # inp w/ clip representation w/ exp -> w/ both
    #X = (multimodal_repr.numpy()*concept_repr.numpy())
    # X_ = multimodal_repr.numpy() # inp w/ clip representation w/o exp -> w/ only input
    # X__ = concept_repr1.numpy() # inp w/o clip representation w/ exp -> w/ only exp

    X_ = np.concatenate([multimodal_repr.numpy(),multimodal_repr.numpy()],axis=-1)
    X__ = np.concatenate([concept_repr1.numpy(),concept_repr1.numpy()],axis=-1)

    print(X.shape)
    print(X_.shape)
    print(X__.shape)
    print(len(y_p))
    #print(0/0)



    kf = KFold(n_splits=5)
    y = np.array(y_p)
    vals = []
    vals_lus = []
    k1,k2,k3 = [],[],[]
    C,S = [],[]
    for train, test in tqdm(kf.split(X)):

        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        X_train_, X_test_, y_train, y_test = X_[train], X_[test], y[train], y[test]
        X_train__, X_test__, y_train, y_test = X__[train], X__[test], y[train], y[test]

        
        clf1 = make_pipeline(StandardScaler(),svm.SVC(probability=True, kernel='rbf', C=2, random_state=42))
       

        clf1 = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(1024, 512), 
                                                    alpha=0.0001, 
                                                    max_iter=10, 
                                                    random_state=42))

        


        clf1.fit(X_train, y_train)
        y_pred = clf1.predict(X_test)

        # clf2 = make_pipeline(StandardScaler(),svm.SVC(probability=True,kernel='rbf', C=2, random_state=42))
        # clf2.fit(X_train_, y_train)
        y_pred_ = clf1.predict(X_test_)

        # clf3 = make_pipeline(StandardScaler(), svm.SVC(probability=True,kernel='rbf', C=2, random_state=42))
        # clf3.fit(X_train__, y_train)
        y_pred__ = clf1.predict(X_test__)


        # y_pred_proba_w_both = clf1.predict_proba(X_test)
        # y_pred_proba_w_inp = clf2.predict_proba(X_test_)
        # y_pred_proba_w_exp = clf3.predict_proba(X_test__)

        y_pred_proba_w_both = clf1.predict_proba(X_test)
        y_pred_proba_w_inp = clf1.predict_proba(X_test_)
        y_pred_proba_w_exp = clf1.predict_proba(X_test__)



        comprehensiveness, sufficiency = 0,0

        counter = 0
        tot_counter = 0
        for pred_proba_w_both, pred_proba_w_inp, pred_proba_w_exp, pred_class in zip(y_pred_proba_w_both,y_pred_proba_w_inp,y_pred_proba_w_exp,y_pred):
            comprehensiveness += (pred_proba_w_both[pred_class] - pred_proba_w_inp[pred_class])
            sufficiency += (pred_proba_w_both[pred_class] - pred_proba_w_exp[pred_class])
            counter+=1

            
            
            try:
                comprehensiveness += (pred_proba_w_both[pred_class] - pred_proba_w_inp[pred_class])
                sufficiency += (pred_proba_w_both[pred_class] - pred_proba_w_exp[pred_class])
                counter+=1
            except:
                tot_counter+=1
                continue
            

        comprehensiveness = comprehensiveness / counter
        sufficiency = sufficiency / counter

        C.append(comprehensiveness)
        S.append(sufficiency)

        print(tot_counter)



        l, nl = 0,0
        us = 0
        for i ,j in zip(y_pred__,y_test):
            if i==j:
                l+=1
            else:
                nl+=1
        a,b=0,0
        for i,j,k,x in zip(y_pred,y_pred_,y_test,y_pred__):
            us+=int(i==k) - int(j==k)
            if k==x:
                a += int(i==k) - int(j==k)
            elif k!=x:
                b += int(i==k) - int(j==k)

        #print('LAS ',.5*(a/l)+.5*(b/nl))
        #print('Leakage Unadjusted Simulatability (LUS)', (us/(l+nl)))
        vals_lus.append(us/(l+nl))
        #print('Comprehensiveness ', comprehensiveness)
        #print('Sufficiency ', sufficiency)
        #print('f1 between prediction of proposed model and SVM w/ both inp and exp {}'.format(f1_score(y_test, y_pred, average='macro')))
        #print('f1 between prediction of proposed model and SVM w/ only inp {}'.format(f1_score(y_test, y_pred_, average='macro')))
        #print('f1 between prediction of proposed model and SVM w/ only exp {}'.format(f1_score(y_test, y_pred__, average='macro')))
        vals.append(.5*(a/l)+.5*(b/nl))
        k1.append(f1_score(y_test, y_pred, average='macro'))
        k2.append(f1_score(y_test, y_pred_, average='macro'))
        k3.append(f1_score(y_test, y_pred__, average='macro'))


    #print('Intra mean ', np.mean(ID), 'Intra std ', np.std(ID))
    #print('Inter mean ', np.mean(ID1), 'Inter std ', np.std(ID1))
    print('LAS mean ', np.mean(vals), 'LAS std ', np.std(vals))
    print('LUS mean ', np.mean(vals_lus), 'LUS std ', np.std(vals_lus))
    print('Comprehensiveness mean ', np.mean(C), 'Comprehensiveness std ', np.std(C))
    print('Sufficiency mean ', np.mean(S), 'Sufficiency std ', np.std(S))
    print('Avg. f1 between prediction of proposed model and SVM w/ both inp and exp {}'.format(np.mean(k1)))
    print('Avg. f1 between prediction of proposed model and SVM w/ only inp {}'.format(np.mean(k2)))
    print('Avg. f1 between prediction of proposed model and SVM w/ only exp {}'.format(np.mean(k3)))

"""


assert int(multimodal_repr.shape[0])==int(concept_repr.shape[0])
assert int(concept_repr.shape[0])==len(y_p)



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_sizes=(1024, 512), num_classes=None, max_epochs=10, lr=0.0001, random_state=42):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.lr = lr
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        # Set random seed
        torch.manual_seed(self.random_state)
        
        # Get number of classes if not provided
        if self.num_classes is None:
            self.num_classes = len(set(y))
        
        # Store classes
        self.classes_ = np.unique(y)
        
        # Create model
        self.model = MultiClassClassifier(self.input_size, self.hidden_sizes, self.num_classes)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            probas = nn.functional.softmax(outputs, dim=1)
        return probas.numpy()

def get_sim(multimodal_repr, concept_repr1, y_p):
    # X = np.concatenate([multimodal_repr.numpy(), concept_repr1.numpy()], axis=-1)
    # X_ = np.concatenate([multimodal_repr.numpy(), multimodal_repr.numpy()], axis=-1)
    # X__ = np.concatenate([concept_repr1.numpy(), concept_repr1.numpy()], axis=-1)

    X = multimodal_repr.numpy() + concept_repr1.numpy()
    X_ = multimodal_repr.numpy()
    X__ = concept_repr1.numpy()

    print(X.shape)
    print(X_.shape)
    print(X__.shape)
    print(len(y_p))

    kf = KFold(n_splits=5)
    y = np.array(y_p)
    vals = []
    vals_lus = []
    k1, k2, k3 = [], [], []
    C, S = [], []
    
    for train, test in tqdm(kf.split(X)):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        X_train_, X_test_, y_train, y_test = X_[train], X_[test], y[train], y[test]
        X_train__, X_test__, y_train, y_test = X__[train], X__[test], y[train], y[test]

        # Create PyTorch classifier with appropriate input size
        input_size = X_train.shape[1]
        clf1 = make_pipeline(
            StandardScaler(),
            PyTorchClassifier(
                input_size=input_size,
                hidden_sizes=(512, 1024),
                max_epochs=50,
                random_state=42,
                num_classes=3129
            )
        )

        clf1.fit(X_train, y_train)
        y_pred = clf1.predict(X_test)
        y_pred_ = clf1.predict(X_test_)
        y_pred__ = clf1.predict(X_test__)

        y_pred_proba_w_both = clf1.predict_proba(X_test)
        y_pred_proba_w_inp = clf1.predict_proba(X_test_)
        y_pred_proba_w_exp = clf1.predict_proba(X_test__)

        comprehensiveness, sufficiency = 0, 0
        counter = 0
        tot_counter = 0
        
        for pred_proba_w_both, pred_proba_w_inp, pred_proba_w_exp, pred_class in zip(y_pred_proba_w_both, y_pred_proba_w_inp, y_pred_proba_w_exp, y_pred):
            comprehensiveness += (pred_proba_w_both[pred_class] - pred_proba_w_inp[pred_class])
            sufficiency += (pred_proba_w_both[pred_class] - pred_proba_w_exp[pred_class])
            counter += 1

        comprehensiveness = comprehensiveness / counter
        sufficiency = sufficiency / counter

        C.append(comprehensiveness)
        S.append(sufficiency)

        print(tot_counter)

        # Rest of the code remains the same
        l, nl = 0, 0
        us = 0
        for i, j in zip(y_pred__, y_test):
            if i == j:
                l += 1
            else:
                nl += 1
        a, b = 0, 0
        for i, j, k, x in zip(y_pred, y_pred_, y_test, y_pred__):
            us += int(i == k) - int(j == k)
            if k == x:
                a += int(i == k) - int(j == k)
            elif k != x:
                b += int(i == k) - int(j == k)

        vals_lus.append(us/(l+nl))
        vals.append(.5*(a/l)+.5*(b/nl))
        k1.append(f1_score(y_test, y_pred, average='macro'))
        k2.append(f1_score(y_test, y_pred_, average='macro'))
        k3.append(f1_score(y_test, y_pred__, average='macro'))

    print('LAS mean ', np.mean(vals), 'LAS std ', np.std(vals))
    print('LUS mean ', np.mean(vals_lus), 'LUS std ', np.std(vals_lus))
    print('Comprehensiveness mean ', np.mean(C), 'Comprehensiveness std ', np.std(C))
    print('Sufficiency mean ', np.mean(S), 'Sufficiency std ', np.std(S))
    print('Avg. f1 between prediction of proposed model and SVM w/ both inp and exp {}'.format(np.mean(k1)))
    print('Avg. f1 between prediction of proposed model and SVM w/ only inp {}'.format(np.mean(k2)))
    print('Avg. f1 between prediction of proposed model and SVM w/ only exp {}'.format(np.mean(k3)))



get_sim(multimodal_repr, concept_repr, y_p)