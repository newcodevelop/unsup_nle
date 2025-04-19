import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, CLIPTextModel, CLIPTokenizer
from PIL import Image
from scipy.spatial.distance import cosine




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
from transformers import AutoImageProcessor, AutoModel

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

            input_ids = batch['input_ids']
            visual_feats = batch['visual_feats']
            visual_pos = batch['visual_pos']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']

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

positives_ = torch.load("./positive.pt")
negatives_ = torch.load("./negative.pt")
anchors_ = torch.load("./anchor.pt")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the projection model
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

# Define the triplet loss
class TripletLoss():
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_loss(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)

        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)

# Function to train the model
def train_projector(anchor_tensor, positive_tensor, negative_tensor, batch_size=8, num_epochs=100):
    # Create dataset and dataloader
    dataset = TensorDataset(anchor_tensor, positive_tensor, negative_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get dimensions from the data
    input_dim = anchor_tensor.shape[1]
    output_dim = positive_tensor.shape[1]

    # Initialize the model, loss function, and optimizer
    proj_model = Projector(input_dim=input_dim, output_dim=output_dim).to('cuda')
    triplet_loss = TripletLoss(margin=1.0)
    optimizer = optim.Adam(proj_model.parameters(), lr=0.00001)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_anchor, batch_positive, batch_negative in dataloader:
            # Forward pass

            projected_anchor = proj_model(batch_anchor.to('cuda'))

            # Compute loss

            pos_dist = torch.sum(torch.pow(projected_anchor - batch_positive.to('cuda'), 2), dim=1)
            neg_dist = torch.sum(torch.pow(projected_anchor - batch_negative.to('cuda'), 2), dim=1)

            # Compute triplet loss
            loss = torch.clamp(pos_dist - neg_dist + 1.0, min=0.0)
            loss = torch.mean(loss)


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

# Example usage

# Sample data - in a real scenario, these would be your actual tensors
# The first dimension (batch size) can be any number
batch_size = 20  # This can be any number
# anchor = torch.randn(batch_size, 768)
# positive = torch.randn(batch_size, 512)
# negative = torch.randn(batch_size, 512)

# Train the model
proj_model = train_projector(
    anchor_tensor=anchors_,
    positive_tensor=positives_,
    negative_tensor=negatives_,
    batch_size=8,  # Process 8 examples at a time
    num_epochs=50
)



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
  # print('cand', candidates)


  clip_inputs = clip_tokenizer(candidates, return_tensors="pt", padding=True)



  with torch.no_grad():
      clip_embeddings = clip_model.get_text_features(input_ids = clip_inputs['input_ids'].to('cuda'), attention_mask = clip_inputs['attention_mask'].to('cuda'))

  # print(clip_embeddings.shape)

  with torch.no_grad():
    target_future_embedding = mlp(clip_embeddings)

  # print(target_future_embedding.shape)
  dist_projected = distance.cosine(target_future_embedding.cpu().squeeze().numpy(), constant_vector.cpu().squeeze().numpy())
  dist_now = distance.cosine(clip_embeddings.cpu().squeeze().numpy(), constant_vector.cpu().squeeze().numpy())

  dist = (dist_projected+dist_now)/2

  if dist>=threshold:
    return True



def beam_search(llava_model, llava_processor, constant_vector, input_text, image, beam_width=5, max_length=15, topk=200):
    # Tokenize input text
    # input_ids = tokenizer.encode(input_text, return_tensors="pt")
    inputs = llava_processor(text=input_text, images=image, return_tensors="pt").to('cuda')

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
                outputs = llava_model(**seq,use_cahce=False)
                logits = outputs.logits[:, -1, :]

            # Get top-k next tokens
            next_token_logits = torch.log_softmax(logits, dim=-1)
            topk_logits, topk_indices = torch.topk(next_token_logits, topk)

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

                if satisfies_constraint_or_promising(llava_processor, new_seq, constant_vector, k=50, threshold=0.9):
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


for p, i in enumerate(dataset):
    text = i['question']
    kk = i['image_id'].split('::')[0].split('://')[-1].split('/')[-1]
    # url = "http://images.cocodataset.org/val2014/COCO_val2014_000000262148.jpg"
    img_path = os.path.join("../val2014/", kk)

    # image = Image.open(requests.get(image_path, stream=True).raw)
    visual_embeds, normalized_boxes = get_visual_embedding(img_path)

    # print(text, image_path)
    # prepare inputs
    # image = image.open(image_path)
    # encoding = processor(image, text, return_tensors="pt")

    inputs = tokenizer(text)
    if args.model.lower()=='lxmert':
            
        inputs.update({
            "visual_feats": torch.squeeze(visual_embeds),
            "visual_pos": torch.squeeze(normalized_boxes),
            "image_path": image_path,
            })
        with torch.no_grad():
            pred_lab, pooled_output = model(batch)

        print('lxmert pooled', pooled_output.shape)
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