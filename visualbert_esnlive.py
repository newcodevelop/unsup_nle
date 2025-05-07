import numpy as np
# import jsonlines
from tqdm import tqdm
import os

# import sys
# sys.path.append('../../exps/transformers/examples/research_projects/visual_bert')
import sys
sys.path.append('/home/dibyanayan/unsup_nle/transformers-research-projects/visual_bert')
import os
print('right file execution')
#os.environ["LD_LIBRARY_PATH"] = "/home1/ekbal_asif/baban/multimodal/cuda/cuda/lib64"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
# from IPython.display import Image, display
import PIL.Image
import io
import torch
import numpy as np

from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
import pandas as pd

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
    visual_embeds = features
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    return visual_embeds,visual_token_type_ids,visual_attention_mask

csv_file   = '/home/dibyanayan/unsup_nle/e-ViL/data/esnlive_train.csv'
image_root = '/home/dibyanayan/unsup_nle/flickr30k_images/flickr30k_images/flickr30k_images'

data = pd.read_csv(csv_file)
data['Name'] = data['Flickr30kID'] 
outliers = []

outliers = []
for names in tqdm(list(data['Name'])):
    if not os.path.exists(os.path.join(image_root, names)):
        outliers.append(names)

data = data[~data['Name'].isin(outliers)]


img_path = os.path.join(image_root, list(data['Name'])[0])
tmp_visual_embeds, tmp_visual_token_type_ids, tmp_visual_attention_mask = get_visual_embedding(img_path)

visual_feats = {}

for _, i in tqdm(data.iterrows()):
    img_path = os.path.join(image_root, i['Name'])
    try:
        visual_embeds, visual_token_type_ids, visual_attention_mask = get_visual_embedding(img_path)
    except:
        visual_embeds, visual_token_type_ids, visual_attention_mask = (
            tmp_visual_embeds, tmp_visual_token_type_ids, tmp_visual_attention_mask
        )
    visual_feats[i['Name']] = (visual_embeds, visual_token_type_ids, visual_attention_mask)



# torch.save(visual_feats, './tensors/vbert_esnlive.pt')
print("Visual features saved to ./tensors/vbert_esnlive.pt")

