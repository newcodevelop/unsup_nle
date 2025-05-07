import numpy as np
import json
from tqdm import tqdm
import os
import torch
import sys
# from processing_image import Preprocess
# from modeling_frcnn import GeneralizedRCNN
from transformers import LxmertTokenizer, LxmertForQuestionAnswering
from PIL import Image
import torchvision
from torchvision import transforms
import requests
from io import BytesIO



# # script.py
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--eval', type=str, required=True)
# args = parser.parse_args()

# print(f"Hi, {args.eval}!")


import sys
sys.path.append('/home/dibyanayan/unsup_nle/transformers-research-projects/visual_bert')


import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

device = 'cuda'

from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config


device = 'cuda'
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.MODEL.device = device
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

frcnn.eval()

def get_visual_embedding(img_paths):
    images, sizes, scales_yx = image_preprocess(img_paths) 
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


# base_dir = '/home/dibyanayan/unsup_nle/train2014'
base_dir = '/home/dibyanayan/unsup_nle/flickr30k_images/flickr30k_images/flickr30k_images'
csv_file = '/home/dibyanayan/unsup_nle/e-ViL/data/esnlive_train.csv'


# kk = os.listdir(base_dir)

# image_paths = []
# for i in kk:
#     if i.endswith('.jpg'):
#         image_paths.append(os.path.join(base_dir, i))

# print(len(image_paths), image_paths[:5])


image_paths = []
with open(csv_file, 'r', encoding='utf‑8') as f:
    next(f)                                
    for line in f:
        flickr_id = line.split(',')[2].strip() 
        img_path  = os.path.join(base_dir, flickr_id)
        image_paths.append(img_path)

print(len(image_paths), image_paths[:5])

visual_feats = {}

count = 0

for img_path in tqdm(image_paths):
    
    try:
        visual_embeds, normalized_boxes = get_visual_embedding(img_path)
        visual_feats[img_path] = (visual_embeds, normalized_boxes)
        
    except:
        count+=1
        continue

print('Number of images not processed {}'.format(count))
torch.save(visual_feats, './tensors/vs_tensors_esnlive.pt')
print("Visual features saved to ./tensors/vs_tensors_esnlive.pt")


