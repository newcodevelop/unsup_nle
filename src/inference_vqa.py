import torch
from torch import nn
from transformers import LlavaProcessor, LlavaForConditionalGeneration, CLIPTextModel, CLIPTokenizer
from PIL import Image
from scipy.spatial.distance import cosine

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

print(f"Model: {args.model}")

#Run checks






if args.model.lower() not in ['visualbert', 'lxmert']:
    raise Exception("Model should be between visualbert, lxmert")


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











from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_id = "llava-hf/llava-1.5-7b-hf"

llava_processor = AutoProcessor.from_pretrained(model_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")


clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')



from transformers import CLIPTokenizer, CLIPModel
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cuda')


from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image



# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


import torch

positives_ = torch.load("./{}_vqav2_5000_positive.pt".format(args.model.lower()))
negatives_ = torch.load("./{}_vqav2_5000_negative.pt".format(args.model.lower()))
anchors_ = torch.load("./{}_vqav2_5000_anchor.pt".format(args.model.lower()))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Define the projection model

"""
class Projector(nn.Module):
    def __init__(self, input_dim=768, output_dim=512):
        super(Projector, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 640),
            nn.ReLU(),
            nn.Linear(640, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class DisentangleProjectorQ(nn.Module):
    
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_Q(x_concepts, y_concepts, batch_size=32, num_epochs=40):
    dataset = TensorDataset(x_concepts, y_concepts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = x_concepts.shape[1]
    model_Q = DisentangleProjectorQ(input_dim, 256).to(device)
    optimizer = optim.Adam(model_Q.parameters(), lr=0.0001)
    margin = 1
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_positive, batch_negative in loader:
            q_x = F.normalize(model_Q(batch_positive.to('cuda')), dim=1)
            q_y = F.normalize(model_Q(batch_negative.to('cuda')), dim=1)
            # Distance between each paired rep
            dist = torch.norm(q_x - q_y, dim=1)
            # Triplet-like: enforce dist >= margin
            loss = F.relu(margin - dist).mean()
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Q] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

    return model_Q

model_Q = train_Q(positives_, negatives_)

# Function to train the model
def train_projector(model_Q, anchor_tensor, positive_tensor, negative_tensor, batch_size=8, num_epochs=100):
    # Create dataset and dataloader


    
    dataset = TensorDataset(anchor_tensor, positive_tensor, negative_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get dimensions from the data
    input_dim = anchor_tensor.shape[1]
    output_dim = positive_tensor.shape[1]

    # Initialize the model, loss function, and optimizer
    proj_model = Projector(input_dim=input_dim, output_dim=256).to('cuda')
    # triplet_loss = TripletLoss(margin=1.0)
    optimizer = optim.Adam(proj_model.parameters(), lr=0.00001)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_anchor, batch_positive, batch_negative in dataloader:
            # Forward pass

            projected_anchor = proj_model(batch_anchor.to('cuda'))

            # Compute loss

            # print(projected_anchor.shape, batch_positive.shape)
            positive = F.normalize(model_Q(batch_positive.to('cuda').cuda()), dim=1) # positive and negative centroids in this space is linearly separable.
            negative = F.normalize(model_Q(batch_negative.to('cuda').cuda()), dim=1)

           


            anchor = F.normalize(projected_anchor, p=2, dim=1)
            # positive = F.normalize(batch_positive.to('cuda'), p=2, dim=1)
            # negative = F.normalize(batch_negative.to('cuda'), p=2, dim=1)


            # pos_cossim = F.cosine_similarity(projected_anchor, batch_positive.to('cuda'), dim=1).mean()
            # neg_cossim = F.cosine_similarity(projected_anchor, batch_negative.to('cuda'), dim=1).mean()

            pos_cossim = F.cosine_similarity(projected_anchor, positive, dim=1).mean()
            neg_cossim = F.cosine_similarity(projected_anchor, negative, dim=1).mean()

            # print(pos_cossim,neg_cossim)

            # print(0/0)

            pos_dist = torch.sum(torch.pow(projected_anchor -  positive, 2), dim=1)
            neg_dist = torch.sum(torch.pow(projected_anchor - negative, 2), dim=1)

            # Compute triplet loss
            loss = torch.clamp(pos_dist - neg_dist + 2.0, min=0.0) 
            loss_cossim = 0.5*(pos_cossim - neg_cossim)
            # loss_cossim = (pos_cossim - neg_cossim)
            

            pos_centroid = torch.mean(positive, dim=0)
            neg_centroid = torch.mean(negative, dim=0)

            centroid_loss = -torch.norm(pos_centroid - neg_centroid, p=2)

            loss = torch.mean(loss)
            # loss += loss_cossim
            # loss = (pos_cossim - neg_cossim)

            # loss = triplet_loss.calc_loss(projected_anchor, batch_positive.to('cuda'), batch_negative.to('cuda'))

            # Backward pass and optimize

            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Print progress
        # if (epoch+1) % 10 == 0:
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    return proj_model



# Function to train the model
def test_projector(model_Q, proj_model, anchor_tensor, positive_tensor, negative_tensor, batch_size=8):
    # Create dataset and dataloader
    dataset = TensorDataset(anchor_tensor, positive_tensor, negative_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get dimensions from the data
    input_dim = anchor_tensor.shape[1]
    output_dim = positive_tensor.shape[1]

    # # Initialize the model, loss function, and optimizer
    # proj_model = Projector(input_dim=input_dim, output_dim=output_dim).to('cuda')
    # triplet_loss = TripletLoss(margin=1.0)
    # optimizer = optim.Adam(proj_model.parameters(), lr=0.00001)
    dist_pos = []
    dist_neg = []

    proj_model.eval()
   

    for batch_anchor, batch_positive, batch_negative in dataloader:
        # Forward pass

        projected_anchor = proj_model(batch_anchor.to('cuda'))

        # Compute loss

        # print(projected_anchor.shape, batch_positive.shape)

        
        anchor = F.normalize(projected_anchor, p=2, dim=1)
        positive = F.normalize(model_Q(batch_positive.to('cuda').cuda()), dim=1) # positive and negative centroids in this space is linearly separable.
        negative = F.normalize(model_Q(batch_negative.to('cuda').cuda()), dim=1)


        pos_cossim = F.cosine_similarity(projected_anchor, positive, dim=1).mean()
        neg_cossim = F.cosine_similarity(projected_anchor, negative, dim=1).mean()

        # print(pos_cossim,neg_cossim)

        # print(0/0)

        pos_dist = torch.sum(torch.pow(projected_anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(projected_anchor - negative, 2), dim=1)

        dist_pos.append(pos_dist)
        dist_neg.append(neg_dist)

           


    return torch.stack(dist_pos).flatten(), torch.stack(dist_neg).flatten()

# Example usage

# Sample data - in a real scenario, these would be your actual tensors
# The first dimension (batch size) can be any number
batch_size = 20  # This can be any number
# anchor = torch.randn(batch_size, 768)
# positive = torch.randn(batch_size, 512)
# negative = torch.randn(batch_size, 512)

# Train the model
proj_model = train_projector(
    model_Q,
    anchor_tensor=anchors_,
    positive_tensor=positives_,
    negative_tensor=negatives_,
    batch_size=8,  # Process 8 examples at a time
    num_epochs=50
)



dist_pos, dist_neg = test_projector(
    model_Q,
    proj_model,
    anchor_tensor=anchors_,
    positive_tensor=positives_,
    negative_tensor=negatives_,
    batch_size=8,  # Process 8 examples at a time
)



print(dist_pos.shape, dist_neg.shape)


# 1) using torch.quantile (PyTorch ≥1.7)
alpha = torch.quantile(dist_pos, 0.95)  # 95th percentile
beta  = torch.quantile(dist_neg, 0.05)   #  5th percentile

print(alpha, beta)

print(0/0)
"""





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Model Definitions
# -----------------------------

class DisentangleProjectorQ(nn.Module):
    """
    Projects concept centroids into a latent space for pairwise separation.
    """
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class ProjectorF(nn.Module):
    """
    Projects anchor representations into the same latent space.
    """
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)

# -----------------------------
# Stage 1: Train Q on paired concept centroids
# -----------------------------

def train_Q(concepts_X, concepts_Y,
            embed_dim=256, margin=1.0,
            batch_size=16, num_epochs=50,
            lr=1e-4, device='cuda'):
    """
    Train Q so that for each pair (x_i, y_i):
      ||Q(x_i) - Q(y_i)|| >= margin

    Returns trained Q.
    """
    dataset = TensorDataset(concepts_X, concepts_Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_Q = DisentangleProjectorQ(concepts_X.size(1), embed_dim).to(device)
    optimizer = optim.Adam(model_Q.parameters(), lr=lr)

    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        epoch_dist = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            q_x = F.normalize(model_Q(x_batch), dim=1)
            q_y = F.normalize(model_Q(y_batch), dim=1)
            dist = torch.norm(q_x - q_y, dim=1)

            # enforce dist >= margin per pair
            loss = F.relu(margin - dist).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dist += dist.mean().item()

        print(f"[Q] Epoch {epoch}/{num_epochs} | Loss: {epoch_loss/len(loader):.4f}" \
              f" | Mean pairwise dist: {epoch_dist/len(loader):.4f}")

    return model_Q

# -----------------------------
# Stage 2: Train F on anchors with paired concept prototypes
# -----------------------------

def train_F(anchors, concepts_X, concepts_Y, model_Q,
            embed_dim=256, margin=0.5,
            batch_size=32, num_epochs=50,
            lr=1e-5, device='cuda'):
    """
    For each example i, trains F so that:
      ||F(anchor_i) - Q(x_i)|| <= ||F(anchor_i) - Q(y_i)|| - margin
    using triplet-like loss.

    anchors: Tensor [N, d_anchor]
    concepts_X, concepts_Y: paired Tensors [N, d_concept]
    model_Q: trained DisentangleProjectorQ
    """
    dataset = TensorDataset(anchors, concepts_X, concepts_Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_F = ProjectorF(anchors.size(1), embed_dim).to(device)
    optimizer = optim.Adam(model_F.parameters(), lr=lr)

    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        for a_batch, x_batch, y_batch in loader:
            a_batch = a_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # forward
            f_a = F.normalize(model_F(a_batch), dim=1)
            with torch.no_grad():
                q_x = F.normalize(model_Q(x_batch), dim=1)
                q_y = F.normalize(model_Q(y_batch), dim=1)

            pos_dist = torch.norm(f_a - q_x, dim=1)
            neg_dist = torch.norm(f_a - q_y, dim=1)

            # triplet loss: want pos_dist + margin <= neg_dist
            loss = F.relu(pos_dist - neg_dist + margin).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[F] Epoch {epoch}/{num_epochs} | Triplet Loss: {epoch_loss/len(loader):.4f}")

    return model_F

# -----------------------------
# Usage Example
# -----------------------------
model_Q = train_Q(positives_, negatives_)
proj_F  = train_F(anchors_, positives_, negatives_, model_Q)

# After training, evaluate pairwise distances:
with torch.no_grad():
    f_out = F.normalize(proj_F(anchors_.to(device)), dim=1)
    q_x  = F.normalize(model_Q(positives_.to(device)), dim=1)
    q_y  = F.normalize(model_Q(negatives_.to(device)), dim=1)
    d_pos = torch.norm(f_out - q_x, dim=1)
    d_neg = torch.norm(f_out - q_y, dim=1)
print('Mean pos dist:', d_pos.mean().item(), 'Mean neg dist:', d_neg.mean().item())



alpha = torch.quantile(d_pos, 0.95)  # 95th percentile
beta  = torch.quantile(d_neg, 0.05)   #  5th percentile

print(alpha, beta)

threshold = (beta-alpha)/2

threshold = threshold.item()










# Function to project new anchors
def project_anchor(new_anchor_batch):
    proj_model.eval()
    with torch.no_grad():
        return proj_model(new_anchor_batch)



max_length = 50
import requests

import datasets
import os


from datasets import load_dataset

# Stream the dataset instead of downloading it completely
dataset = load_dataset("Graphcore/vqa", split="validation", streaming=True, trust_remote_code=True)

# Take only the first N examples
dataset = dataset.take(5000)


from tqdm import tqdm

# Step 5: Train a simple MLP
import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)



mlp = MLP()

PATH = './mlp_model.pt'
mlp.load_state_dict(torch.load(PATH, weights_only=True))
mlp.to('cuda')
mlp.eval()

from scipy.spatial import distance


def satisfies_constraint_or_promising(llava_processor, new_seq, constant_vector, k=50, threshold=0.995):
  new_input_ids = new_seq['input_ids']
  candidates = llava_processor.tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)[-1].split('ASSISTANT:')[-1]
  #   print('cand', candidates)


  clip_inputs = clip_tokenizer(candidates, return_tensors="pt", padding=True)



  with torch.no_grad():
      clip_embeddings = clip_model.get_text_features(input_ids = clip_inputs['input_ids'].to('cuda'), attention_mask = clip_inputs['attention_mask'].to('cuda'))

  # print(clip_embeddings.shape)

  with torch.no_grad():
    target_future_embedding = mlp(clip_embeddings)

  # print(target_future_embedding.shape)
  #   print(F.cosine_similarity(target_future_embedding, constant_vector, dim=-1))
  dist_projected = distance.cosine(target_future_embedding.cpu().squeeze().numpy(), constant_vector.cpu().squeeze().numpy())
  #   print(dist_projected)
  dist_now = distance.cosine(clip_embeddings.cpu().squeeze().numpy(), constant_vector.cpu().squeeze().numpy())

  dist = (dist_projected+dist_now)/2

  if dist>=threshold:
    return True

llava_tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")








def beam_search_1(llava_model, llava_processor, constant_vector, input_text, image, beam_width=4, max_length=20, topk=800, threshold=0.9):
    # Tokenize input text
    # input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # inputs = llava_processor(input_text, images=image, return_tensors="pt").to('cuda')
    inputs = llava_processor(text=input_text, images=image, padding=True, truncation=True, return_tensors="pt").to('cuda')
   

    # print('*'*10)
    # for i in inputs:
    #     print(inputs[i].shape)


    # print(0/0)
    num_batches = int(inputs['input_ids'].shape[0])

    batch_beams={}

    # Initialize beam with the input sequence and score 0.0
    for i in range(num_batches):
        d = {"input_ids": inputs["input_ids"][i,:].unsqueeze(0), "attention_mask": inputs["attention_mask"][i,:].unsqueeze(0), "pixel_values": inputs["pixel_values"][i,:].unsqueeze(0)}
    
        # batch_beams[i] = [(d, 0.0)]*beam_width
        batch_beams[i] = [(d, 0.0)]


    # batch_beams = {i: (inp, 0) for i in inputs[o]}

    # beam = [(inputs, 0.0)]

    dropped = {}

    prelim_seq_len = batch_beams[0][0][0]['input_ids'].shape[-1]

    for curr_len in tqdm(range(max_length)):

        batch_seq_all = {'input_ids': [], 'attention_mask': [], 'pixel_values': []}

        dimensions = []

        for cnt, beam in enumerate(list(batch_beams)):

            # cnt can run from 1 to batch_size

            all_candidates = []
            batch_seq_input_ids = []
            batch_seq_attn_mask = []
            batch_seq_pixel_values = []


            beam_content = batch_beams[beam]

            

            for seq, score in beam_content:
                # Check if sequence is complete

                # if seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id:
                #     all_candidates.append((seq, score))
                #     continue

                batch_seq_input_ids.append(seq["input_ids"])
                batch_seq_attn_mask.append(seq["attention_mask"])
                batch_seq_pixel_values.append(seq["pixel_values"])

                

            
            
           
            batch_seq_input_ids = torch.cat(batch_seq_input_ids, dim=0) # bw_cnt, max_len
            batch_seq_attn_mask = torch.cat(batch_seq_attn_mask, dim=0)
            batch_seq_pixel_values = torch.cat(batch_seq_pixel_values, dim=0)

            dimensions.append(batch_seq_input_ids.shape[0])

            print(batch_seq_input_ids.shape)

        
            # batch_seq = {"input_ids":batch_seq_input_ids , "attention_mask": batch_seq_attn_mask, "pixel_values": batch_seq_pixel_values}

            batch_seq_all['input_ids'].append(batch_seq_input_ids) #sizes of [bw1, max_len], [bw2, max_len]; bw1, bw2 are corresponding beam lengths, they might not be the same. bw1!=bw2
            batch_seq_all['attention_mask'].append(batch_seq_attn_mask)
            batch_seq_all['pixel_values'].append(batch_seq_pixel_values)
        

        batch_seq_all['input_ids'] = torch.cat(batch_seq_all['input_ids'], dim=0) # (BS, beam_width, max_len)
        batch_seq_all['attention_mask'] = torch.cat(batch_seq_all['attention_mask'], dim=0)
        batch_seq_all['pixel_values'] = torch.cat(batch_seq_all['pixel_values'], dim=0)

        for i in batch_seq_all:
            print(i, batch_seq_all[i].shape)

        # B, N = batch_seq_all['input_ids'].shape[:2]

        # batch_seq_all['input_ids'] = batch_seq_all['input_ids'].reshape(B * N, -1)
        # batch_seq_all['attention_mask'] = batch_seq_all['attention_mask'].reshape(B * N, -1)
        # batch_seq_all['pixel_values'] = batch_seq_all['pixel_values'].reshape(B * N, *batch_seq_all['pixel_values'].shape[2:])

        with torch.no_grad():
            outputs = llava_model(**batch_seq_all,use_cache=True)
            logits = outputs.logits[:, -1, :]

        # Get top-k next tokens
        next_token_logits = torch.log_softmax(logits, dim=-1)
        topk_logits, topk_indices = torch.topk(next_token_logits, topk)

        topk_indices_splits = list(torch.split(topk_indices, dimensions, dim=0))
        topk_logits_splits = list(torch.split(topk_logits, dimensions, dim=0))

        for i,j in zip(topk_indices_splits, topk_logits_splits):
            print(i.shape, j.shape)
        print(len(topk_indices_splits))


        for cnt, beam in enumerate(list(batch_beams)):
            beam_content = batch_beams[beam]
            all_candidates = []
            for width, (seq, score) in enumerate(beam_content):
                token_id = topk_indices_splits[cnt][width, :].unsqueeze(0) # (1,200)
                token_score = topk_logits_splits[cnt][width, :] # 200 dim list

                new_score = (score * (seq["input_ids"].shape[1] - 1) + token_score) / (seq["input_ids"].shape[1])

                seq_len = seq["input_ids"].shape[1]

                t1 = seq["input_ids"].expand(topk,-1,-1) # seq["input_ids"] len is seq_len, shape = (1,seq_len)
                t2 = token_id.view(topk,1,1)
                t3 = torch.cat([t1,t2],dim=-1).squeeze() # (200,seq_len+1)

                print(t3.shape)

                candidates = llava_processor.tokenizer.batch_decode(t3, skip_special_tokens=True)
                candidates = list(map(lambda x: x.split('ASSISTANT:')[-1].strip(), candidates))

                print('cand', candidates[:10], len(candidates))


                clip_inputs = clip_tokenizer(candidates, return_tensors="pt", padding=True)



                with torch.no_grad():
                    clip_embeddings = clip_model.get_text_features(input_ids = clip_inputs['input_ids'].to('cuda'), attention_mask = clip_inputs['attention_mask'].to('cuda'))

                print(clip_embeddings.shape)

                

                with torch.no_grad():
                    target_future_embedding = mlp(clip_embeddings)
                print(target_future_embedding.shape)
                cv = constant_vector[cnt,:].unsqueeze(0).expand(topk,-1)
                # similarities_expected = F.cosine_similarity(target_future_embedding, cv, dim=1)
                # similarities_now = F.cosine_similarity(clip_embeddings, cv, dim=1)
                # print(similarities_expected.shape, similarities_now.shape)
                # similarities = 0.9*similarities_expected + 0.1*similarities_now
                # print(similarities_expected[:10], similarities_now[:10], similarities[:10])

                # threshold_mean = 0.9*similarities_expected.mean() + 0.1*similarities_now.mean()
                # threshold_std = 0.9*similarities_expected.std() + 0.1*similarities_now.std()

                # threshold_expected = similarities_expected.mean() + 3*similarities_expected.std()
                # threshold_now = similarities_now.mean() + 3*similarities_now.std()

                # threshold = 0.9*threshold_expected + 0.1*threshold_now

                # threshold = threshold_mean + 2*threshold_std

                target_future_embedding = F.normalize(model_Q(target_future_embedding.to(device)), dim=1)
                clip_embeddings = F.normalize(model_Q(clip_embeddings.to(device)), dim=1)

                print(cv.shape,target_future_embedding.shape, clip_embeddings.shape )

                dist1 = torch.norm(cv - target_future_embedding, dim=1)
                dist2 = torch.norm(cv - clip_embeddings, dim=1)



                


                dist = (dist1+dist2)/2

                

                print(dist1[:10], dist2[:10], dist[:10], threshold)

               

                # threshold_mean = 0.9*dist1.mean() + 0.1*dist2.mean()
                # threshold_std = 0.9*dist1.std() + 0.1*dist2.std()

                # threshold = dist.mean() - 2*dist.std()

                # print(threshold.item())

                dist_sorted, indices_sort = dist.sort(dim=-1)

                # print(dist[:10], dist_sorted[:10])

                # # percentage = 0.1 #only select top 10% of the topk tokens based on their closeness to the projected multimodal embedding (the constraint)
                percentage = 0.2
                num_vals = int(percentage*topk)-1

                threshold = dist_sorted[num_vals].item()



                # print(dist[:20])

                # print(0/0)

                
                num_examples = 0
                for counter, i in enumerate(list(dist.detach().cpu().numpy())):
                    
                    if i.item()<=threshold:
                        num_examples +=1
                
                print('number of such examples {}'.format(num_examples))


                for counter, i in enumerate(list(dist.detach().cpu().numpy())):
                    
                    if i.item()<=threshold: # cosine similarity must be positive (>=epsilon)
                        new_input_ids = t3[counter,:].unsqueeze(0)
                        attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long).to('cuda')
                        new_score = (score * (seq_len - 1) + token_score[counter].item()) / (seq_len)
                        
                        new_seq = {"input_ids": new_input_ids, "attention_mask": attention_mask, "pixel_values": seq["pixel_values"]}
                        all_candidates.append((new_seq, new_score))
                    else:
                        if (t3[counter,:].unsqueeze(0).shape[1]-prelim_seq_len)==1: # give a chance for first tokens (explore all possibility for the first token)
                            new_input_ids = t3[counter,:].unsqueeze(0)
                            attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long).to('cuda')
                            new_score = (score * (seq_len - 1) + token_score[counter].item()) / (seq_len)
                            
                            new_seq = {"input_ids": new_input_ids, "attention_mask": attention_mask, "pixel_values": seq["pixel_values"]}
                            all_candidates.append((new_seq, new_score))
                        else:
                            pass


            # Select top beam_width candidates
            if len(all_candidates)==0:
                # means for this beam state, none of the continuation is working, so drop the beam state 
                
                dropped[beam] = beam_content
                del batch_beams[beam]
                continue

            beam_content = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # # Check if all sequences are complete
            # if all(seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id for seq, _ in beam_content):
            #     break

            batch_beams[beam] = beam_content








       
        """



        # combine BS, beam width and make inference and then split again

        # for cnt, beam in enumerate(batch_beams):
        for cnt, beam in enumerate(list(batch_beams)):
            beam_content = batch_beams[beam]
            all_candidates = []
            for width, (seq, score) in enumerate(beam_content):
                # all_candidates.append((seq, score))
                # if seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id:
                #     all_candidates.append((seq, score))
                #     continue
                
                for i in range(topk):

                    token_id = topk_indices_splits[cnt][width, i].unsqueeze(0).unsqueeze(0)
                    token_score = topk_logits_splits[cnt][width, i].item()

                    # token_id = topk_indices[cnt, width, i].unsqueeze(0).unsqueeze(0)
                    # token_score = topk_logits[cnt, width, i].item()
                    # Create new sequence by appending the token

                    new_input_ids = torch.cat((seq["input_ids"],torch.tensor([token_id]).unsqueeze(0).to('cuda')),dim=-1)
                    attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long).to('cuda')
                    new_seq = {"input_ids": new_input_ids, "attention_mask": attention_mask, "pixel_values": seq["pixel_values"]}

                    # Update score (normalize by length to avoid bias toward shorter sequences)
                    new_score = (score * (seq["input_ids"].shape[1] - 1) + token_score) / (seq["input_ids"].shape[1])
                    if satisfies_constraint_or_promising(llava_processor, new_seq, constant_vector[cnt,:], k=50, threshold=threshold):
                        all_candidates.append((new_seq, new_score))
                    print(0/0)
            # Select top beam_width candidates
            if len(all_candidates)==0:
                # means for this beam state, none of the continuation is working, so drop the beam state 
                del batch_beams[beam]
                dropped[beam] = all_candidates
                continue

            beam_content = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # # Check if all sequences are complete
            # if all(seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id for seq, _ in beam_content):
            #     break

            batch_beams[beam] = beam_content

        """
        
    for i in batch_beams:
        b = batch_beams[i]
        print(i, llava_processor.tokenizer.batch_decode(b[0][0]['input_ids'], skip_special_tokens=True))
    for i in dropped:
        b = dropped[i]
        print(i, llava_processor.tokenizer.batch_decode(b[0][0]['input_ids'], skip_special_tokens=True))

    print(0/0)































def beam_search(llava_model, llava_processor, constant_vector, input_text, image, beam_width=5, max_length=15, topk=200, threshold=0.9):
    # Tokenize input text
    # input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # inputs = llava_processor(input_text, images=image, return_tensors="pt").to('cuda')
    inputs = llava_processor(text=input_text, images=image, padding=True, truncation=True, return_tensors="pt").to('cuda')
   

    # print('*'*10)
    # for i in inputs:
    #     print(inputs[i].shape)


    # print(0/0)

    # Initialize beam with the input sequence and score 0.0
    beam = [(inputs, 0.0)]

    for curr_len in tqdm(range(max_length)):
        all_candidates = []


        # print(f'At current length of {curr_len} the beam composition is')
        # for i in beam:
        #   print(llava_processor.tokenizer.batch_decode(i[0]['input_ids'], skip_special_tokens=True), i[1])


        # Process each sequence in the current beam
        for seq, score in beam:
            # Check if sequence is complete

            if seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id:
                all_candidates.append((seq, score))
                continue

            # Get logits for next token
            with torch.no_grad():
                outputs = llava_model(**seq,use_cache=True)
                logits = outputs.logits[:, -1, :]

            # Get top-k next tokens
            next_token_logits = torch.log_softmax(logits, dim=-1)
            topk_logits, topk_indices = torch.topk(next_token_logits, topk)

            print(topk_logits.shape, topk_indices.shape)

            print(topk_indices[:,:10])

            print(0/0)

         

            # Create new candidate sequences
            for i in range(topk):
            # for i in range(beam_width):
                token_id = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                token_score = topk_logits[0, i].item()

                # Create new sequence by appending the token

                new_input_ids = torch.cat((seq["input_ids"],torch.tensor([token_id]).unsqueeze(0).to('cuda')),dim=-1)
                attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long).to('cuda')
                new_seq = {"input_ids": new_input_ids, "attention_mask": attention_mask, "pixel_values": seq["pixel_values"]}

                # Update score (normalize by length to avoid bias toward shorter sequences)
                new_score = (score * (seq["input_ids"].shape[1] - 1) + token_score) / (seq["input_ids"].shape[1])

                if satisfies_constraint_or_promising(llava_processor, new_seq, constant_vector, k=50, threshold=threshold):
                  all_candidates.append((new_seq, new_score))

        # Select top beam_width candidates
        beam = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Check if all sequences are complete
        if all(seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id for seq, _ in beam):
            break

    # Return the sequence with highest score
    # best_seq = beam[0][0]
    # return llava_processor.tokenizer.batch_decode(best_seq[0]["input_ids"], skip_special_tokens=True)
    return beam

def batched_beam_search(llava_model, llava_processor, constant_vector, input_text, image, 
                        beam_width=5, max_length=15, topk=200, threshold=0.9, batch_size=8):
    # Prepare initial inputs
    inputs = llava_processor(text=input_text, images=image, padding=True, truncation=True, return_tensors="pt").to('cuda')
    
    # Initialize beam with the input sequence and score 0.0
    beam = [(inputs, 0.0)]

    

    for curr_len in tqdm(range(max_length)):
        all_candidates = []
        
        # Check if beam is empty or all sequences complete
        if not beam or all(seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id for seq, _ in beam):
            break
            
        # Separate complete and incomplete sequences
        complete_candidates = [(seq, score) for seq, score in beam 
                              if seq["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id]
        
     
        incomplete_sequences = [(seq, score) for seq, score in beam 
                               if seq["input_ids"][0, -1].item() != llava_processor.tokenizer.eos_token_id]

        
        # Process all incomplete sequences in batches
        for batch_start in range(0, len(incomplete_sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(incomplete_sequences))
            batch = incomplete_sequences[batch_start:batch_end]
            
            # Prepare batch inputs
            batch_input_ids = torch.cat([seq["input_ids"] for seq, _ in batch], dim=0)
            batch_attention_mask = torch.cat([seq["attention_mask"] for seq, _ in batch], dim=0)
            batch_pixel_values = torch.cat([seq["pixel_values"] for seq, _ in batch], dim=0)
            
            batch_inputs = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "pixel_values": batch_pixel_values
            }
            
            # Get logits for next tokens in batch
            with torch.no_grad():
                outputs = llava_model(**batch_inputs, use_cache=False)
                batch_logits = outputs.logits[:, -1, :]
            
            # Process each sequence in the batch
             
            for i, (seq, score) in enumerate(batch):
                logits = batch_logits[i].unsqueeze(0)  # Add batch dimension back
                
                # Get top-k next tokens
                next_token_logits = torch.log_softmax(logits, dim=-1)
                topk_logits, topk_indices = torch.topk(next_token_logits, topk)
                print([topk_indices[0, j].item() for j in range(10)])
                
                # Create new candidate sequences
                for j in range(topk):
                    token_id = topk_indices[0, j].item()
                    token_score = topk_logits[0, j].item()
                    
                    # Create new sequence by appending the token
                    print(seq["input_ids"])
                    print(seq["input_ids"].shape)
                    print(torch.tensor([[token_id]]).shape)
                    new_input_ids = torch.cat([seq["input_ids"], 
                                              torch.tensor([[token_id]], device='cuda')], dim=1)
                    attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long, device='cuda')
                    
                    new_seq = {
                        "input_ids": new_input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": seq["pixel_values"]
                    }
                    
                    # Update score (normalize by length to avoid bias toward shorter sequences)
                    new_score = (score * (seq["input_ids"].shape[1] - 1) + token_score) / new_input_ids.shape[1]
                    
                    # Check constraint
                    if satisfies_constraint_or_promising(llava_processor, new_seq, constant_vector, k=50, threshold=threshold):
                        all_candidates.append((new_seq, new_score))
        
        # Add complete candidates back
        all_candidates.extend(complete_candidates)
        
        # Select top beam_width candidates
        beam = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return beam











# def batched_beam_search(llava_model, llava_processor, constant_vector, input_text, image, 
#                         beam_width=5, max_length=15, topk=200, threshold=0.9, batch_size=8):
#     # Prepare initial inputs
#     inputs = llava_processor(text=input_text, images=image, padding=True, truncation=True, return_tensors="pt").to('cuda')
    
#     # Get the initial batch size
#     initial_batch_size = inputs["input_ids"].shape[0]
    
#     # Initialize beam for each batch item
#     beams = []
#     for i in range(initial_batch_size):
#         # Extract the i-th item from the batch while preserving the batch dimension
#         single_item = {}
#         for key, value in inputs.items():
#             if isinstance(value, torch.Tensor):
#                 single_item[key] = value[i:i+1]  # Keep batch dimension
#             else:
#                 single_item[key] = value
        
#         # Initialize beam for this batch item
#         beams.append([(single_item, 0.0)])
    
#     for curr_len in tqdm(range(max_length)):
#         # Check if all beams are complete
#         all_complete = True
#         for beam in beams:
#             if not all(seq_dict["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id for seq_dict, _ in beam):
#                 all_complete = False
#                 break
        
#         if all_complete:
#             break
        
#         # Collect all sequences from all beams that need processing
#         all_sequences = []
#         sequence_origins = []  # Track which beam each sequence came from
        
#         for batch_idx, beam in enumerate(beams):
#             for seq_dict, score in beam:
#                 if seq_dict["input_ids"][0, -1].item() != llava_processor.tokenizer.eos_token_id:
#                     all_sequences.append((seq_dict, score))
#                     sequence_origins.append(batch_idx)
        
#         # Process sequences in batches
#         new_candidates_by_beam = [[] for _ in range(initial_batch_size)]
        
#         for batch_start in range(0, len(all_sequences), batch_size):
#             batch_end = min(batch_start + batch_size, len(all_sequences))
#             batch_sequences = all_sequences[batch_start:batch_end]
#             batch_origins = sequence_origins[batch_start:batch_end]
            
#             # Prepare batch inputs
#             batch_inputs = {
#                 "input_ids": torch.cat([seq["input_ids"] for seq, _ in batch_sequences], dim=0),
#                 "attention_mask": torch.cat([seq["attention_mask"] for seq, _ in batch_sequences], dim=0),
#                 "pixel_values": torch.cat([seq["pixel_values"] for seq, _ in batch_sequences], dim=0)
#             }
            

#             # Get logits for all sequences in one forward pass
#             with torch.no_grad():
#                 outputs = llava_model(**batch_inputs, use_cache=False)
#                 batch_logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            
#             # Process each sequence's logits
#             for i, ((seq_dict, score), beam_idx) in enumerate(zip(batch_sequences, batch_origins)):
#                 logits = batch_logits[i].unsqueeze(0)  # Add batch dimension back
                
#                 # Get top-k next tokens
#                 next_token_logits = torch.log_softmax(logits, dim=-1)
#                 topk_logits, topk_indices = torch.topk(next_token_logits, topk)
#                 print([topk_indices[0, j].item() for j in range(10)])
                    

                
#                 # Create candidates for this sequence
#                 for j in range(topk):
#                     token_id = topk_indices[0, j].item()
#                     token_score = topk_logits[0, j].item()
                    
#                     # Create new sequence
#                     new_input_ids = torch.cat([
#                         seq_dict["input_ids"],
#                         torch.tensor([[token_id]], device='cuda')
#                     ], dim=1)
                    
#                     new_attention_mask = torch.ones_like(new_input_ids)
                    
#                     new_seq_dict = {
#                         "input_ids": new_input_ids,
#                         "attention_mask": new_attention_mask,
#                         "pixel_values": seq_dict["pixel_values"]
#                     }
                    
#                     for i in new_seq_dict:
#                         print(new_seq_dict[i].shape)
#                     # Update score
#                     new_score = (score * (seq_dict["input_ids"].shape[1] - 1) + token_score) / new_input_ids.shape[1]
                    
#                     # Check constraint
#                     if satisfies_constraint_or_promising(llava_processor, new_seq_dict, constant_vector, threshold):
#                         new_candidates_by_beam[beam_idx].append((new_seq_dict, new_score))
        
#         # Update all beams with new candidates
#         for batch_idx in range(initial_batch_size):
#             # Get complete sequences from current beam
#             complete_candidates = [(seq_dict, score) for seq_dict, score in beams[batch_idx] 
#                                   if seq_dict["input_ids"][0, -1].item() == llava_processor.tokenizer.eos_token_id]
            
#             # Combine with new candidates
#             all_candidates = complete_candidates + new_candidates_by_beam[batch_idx]
            
#             # Update beam with top candidates
#             beams[batch_idx] = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
#     # Return the best sequence for each batch item
#     results = []
#     for batch_idx in range(initial_batch_size):
#         results.append(beams[batch_idx][0] if beams[batch_idx] else None)
    
#     return results























dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)


from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer





BATCH_SIZE = 16  # Change as needed

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

for batch in dataloader:
    print(batch)
    inputs = batch[0]
    for i in inputs:
        print(inputs[i].shape)
    with torch.no_grad():
        pred_lab, pooled_output = model(**inputs)
    print(batch[1])
    print(pred_lab, pooled_output.shape)
    pooled_output = pooled_output.squeeze()

    prefix_sets = []

    for idx,txt in zip(pred_lab.squeeze().detach().cpu().numpy(), batch[1]['texts']):

        prefix = f"""USER: <image>\nGiven the input (a question and an image) and the generated answer, generate an explanation behind the answer.\n Question: {txt} Answer: {config.id2label[idx]}. \nASSISTANT: The answer to the question "{txt}" is "{config.id2label[idx]}" because"""
        prefix_sets.append(prefix)

    # print("Predicted answer:", model.config.id2label[idx])
    print('prefix', prefix_sets)
    with torch.no_grad():
        proj_F.eval()
        f_out = F.normalize(proj_F(pooled_output.to('cuda')), dim=1)
        
    
    print(f_out.shape)

    b = beam_search_1(llava_model, llava_processor, f_out, prefix_sets, batch[1]['images'], threshold=0.92)

    # b = batched_beam_search(llava_model, llava_processor, constant_vector, prefix_sets, batch[1]['images'], threshold=0.9)
    break

print(0/0)




for p, i in enumerate(dataset):
    text = i['question']
    kk = i['image_id'].split('::')[0].split('://')[-1].split('/')[-1]
    # url = "http://images.cocodataset.org/val2014/COCO_val2014_000000262148.jpg"
    img_path = os.path.join("../val2014/", kk)

    

    image = Image.open(img_path)
    

    # image = Image.open(requests.get(image_path, stream=True).raw)
    visual_embeds, normalized_boxes = get_visual_embedding(img_path)

    # print(text, image_path)
    # prepare inputs
    # image = image.open(image_path)
    # encoding = processor(image, text, return_tensors="pt")

    inputs = tokenizer(text, return_tensors='pt')
    inputs['input_ids'] = inputs['input_ids'].to('cuda')
    inputs['attention_mask'] = inputs['attention_mask'].to('cuda')
    inputs['token_type_ids'] = inputs['token_type_ids'].to('cuda')
    print(inputs)
    if args.model.lower()=='lxmert':
            
        inputs.update({
            "visual_feats": visual_embeds.to('cuda'),
            "visual_pos": normalized_boxes.to('cuda'),
            "image_path": img_path,
            })

        # for i in inputs:
        #     print(i, inputs[i].shape)

        with torch.no_grad():
            pred_lab, pooled_output = model(inputs)

        idx = pred_lab[0].item()


        # cf_prefix = f"USER: <image>\nGiven the input (a question and an image) and the generated answer, change the question minimally in such a way that the generated answer should change. In one word, generate a counterfactual question by adding an adjective or adverb. \n Question: {text} Answer: {config.id2label[idx]}.\nASSISTANT:"


        # llava_inputs = llava_processor(text=cf_prefix, images=image, return_tensors="pt").to('cuda')

        # kk = llava_model.generate(**llava_inputs)

        # print('kk',llava_processor.batch_decode(kk, skip_special_tokens=True))



        prefix = f"USER: <image>\nGiven the input (a question and an image) and the generated answer, generate an explanation behind the answer.\n Question: {text} Answer: {config.id2label[idx]}. Explanation: Because\nASSISTANT:"

        # print("Predicted answer:", model.config.id2label[idx])
        print('prefix', prefix)
        with torch.no_grad():
            proj_model.eval()
            constant_vector = proj_model(pooled_output.squeeze(dim=1).to('cuda'))

        faithful = {0.92: [], 0.93:[], 0.94:[]}

        for threshold in faithful:
            b = batched_beam_search(llava_model, llava_processor, constant_vector, prefix, image, threshold=threshold) #generate a faithful explanation
            print(llava_processor.tokenizer.batch_decode(b[0][0]['input_ids'], skip_special_tokens=True))
            print(0/0)
            gen_faithful_exp = llava_processor.tokenizer.batch_decode(b[0][0]['input_ids'], skip_special_tokens=True)
            faithful[threshold].append(gen_faithful_exp)

        b = beam_search(llava_model, llava_processor, constant_vector, prefix, image, threshold=0.0) #generate a plausible explanation
        gen_plausible_exp = llava_processor.tokenizer.batch_decode(b[0][0]['input_ids'], skip_special_tokens=True)
        
        print("Plausible {}".format(gen_plausible_exp))

        print(faithful)

        # print(f'At current length of the beam composition is')
        # for i in b:
        #     print(llava_processor.tokenizer.batch_decode(i[0]['input_ids'], skip_special_tokens=True), i[1])

        # print('*****')

        # print('lxmert pooled', pooled_output.shape)


    elif args.model.lower()=='visualbert':

        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)


        inputs.update({
        "visual_embeds": torch.squeeze(visual_embeds),
        "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
        "visual_attention_mask": torch.squeeze(visual_attention_mask),
        "image_path": image_path,
        })

        index_to_gather = inputs['attention_mask'].sum(1) - 2  # as in original code

        with torch.no_grad():
            outputs = model.visual_bert(
            **batch
            )

        sequence_output = outputs[0]

        # TO-CHECK: From the original code
        index_to_gather = (
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        )
        pooled_output_ = torch.gather(sequence_output, 1, index_to_gather)

        pooled_output = model.dropout(pooled_output_)
        logits = model.cls(pooled_output)
        reshaped_logits = logits.view(-1, model.num_labels)
        pred_lab = reshaped_logits.argmax(dim=-1).item()
        print('visualbert pooled', pooled_output_.shape)

    print(0/0)



    outputs = model.vilt(**encoding).pooler_output

    logits = model.classifier(outputs)
    idx = logits.argmax(-1).item()


    prefix = f"USER: <image>\nGiven the input (a question and an image) and the generated answer, generate an explanation behind the answer.\n Question: {text} Answer: {model.config.id2label[idx]}. Explanation: Because\nASSISTANT:"

    # print("Predicted answer:", model.config.id2label[idx])

    with torch.no_grad():
      proj_model.eval()
      constant_vector = proj_model(outputs.to('cuda'))

    b = beam_search(llava_model, llava_processor, constant_vector, prefix, image)

    print(f'At current length of the beam composition is')
    for i in b:
      print(llava_processor.tokenizer.batch_decode(i[0]['input_ids'], skip_special_tokens=True), i[1])

    print('*****')




    if p==100:
        break