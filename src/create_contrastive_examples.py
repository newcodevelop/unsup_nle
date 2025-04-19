import argparse
import torch
import json
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num_examples', type=int, required=True)
args = parser.parse_args()

print(f"Dataset: {args.dataset}, Model: {args.model}")

#Run checks
if args.dataset.lower() not in ['esnlive', 'vqav2']:
    raise Exception("Dataset should be between esnlive, vqav2")

if args.model.lower() not in ['visualbert', 'lxmert']:
    raise Exception("Model should be between visualbert, lxmert")

device = 'cuda'




# if args.dataset.lower() == 'vqav2':
#     kk = 90
# print(kk)

# print(0/0)






import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)
# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)


def get_clip_embeddings(model, processor, image_path, text):
    # Load the CLIP model and processor
   
    image = Image.open(image_path).convert("RGB")
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Get normalized embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
      
        similarity_score = (100.0 * (text_embeds @ image_embeds.T)).item()
    
    return {
        "image_embedding": image_embeds.cpu().numpy(),
        "text_embedding": text_embeds.cpu().numpy(),
        "similarity_score": similarity_score
    }






















all_data = []



if args.dataset=='vqav2':
    
    vs_feat = torch.load('../tensors/vs_tensors_train2014.pt')


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

        def __init__(self, questions, annotations, tokenizer, model_type):
            self.questions = questions
            self.annotations = annotations
            # self.processor = processor
            self.tokenizer = tokenizer
            self.max_len = 128
            self.model_type = model_type

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
            
       
            visual_embeds, normalized_boxes = vs_feat[image_path]

            if self.model_type=='lxmert':
            
                inputs.update({
                    "visual_feats": torch.squeeze(visual_embeds),
                    "visual_pos": torch.squeeze(normalized_boxes),
                    "image_path": image_path,
                    })
            elif self.model_type=='visualbert':

                visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        
                inputs.update({
                "visual_embeds": torch.squeeze(visual_embeds),
                "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
                "visual_attention_mask": torch.squeeze(visual_attention_mask),
                "image_path": image_path,
                })
        
            for k,v in inputs.items():
                if k!="image_path":
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

    from transformers import AutoTokenizer, VisualBertForQuestionAnswering, LxmertModel
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    dataset = VQADataset(questions=questions,
                        annotations=annotations,
                        tokenizer=tokenizer,
                        model_type = args.model.lower())

    
    image_paths = [id_to_filename[annotation['image_id']] for annotation in annotations]
    print(image_paths[:10])
    
    for p, i in tqdm(enumerate(image_paths[:args.num_examples])):
        text = questions[p]['question']
        image_path = image_paths[p]
        # print(text, image_path)
        results = get_clip_embeddings(clip_model, clip_processor, image_path, text)
        all_data.append(results['image_embedding'] + results['text_embedding'])

    print('all data printed')
    # print(0/0)

    for i in dataset:
        print(i)
        break


all_data_y = []

if args.model.lower()=='lxmert' and args.dataset.lower()=='vqav2':
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
    config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    for p,i in enumerate(dataset):
        batch = {'input_ids': i['input_ids'].unsqueeze(0).to('cuda'), 'visual_feats': i['visual_feats'].unsqueeze(0).to('cuda'), 
        'visual_pos': i['visual_pos'].unsqueeze(0).to('cuda'), 'attention_mask': i['attention_mask'].unsqueeze(0).to('cuda'),
        'token_type_ids': i['token_type_ids'].unsqueeze(0).to('cuda')}



        with torch.no_grad():
            pred_lab, pooled_output = model(batch)
            all_data_y.append(pooled_output.cpu())
            pred_lab = pred_lab[0].item()
            true_lab = i['labels'].argmax().item()

            # print(pred_lab, true_lab)
            
            # print(tokenizer.batch_decode(i['input_ids'].unsqueeze(0), skip_special_tokens=True))
            # print(config.id2label[pred_lab], config.id2label[true_lab])
            # print('*'*10)

        if p==args.num_examples-1:
            break




if args.model.lower()=='visualbert' and args.dataset.lower()=='vqav2':
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
    model.load_state_dict(torch.load('../models/vb_vqav2.pt', weights_only=True))
    model.eval()
    model.to(device)
    config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    for p,i in enumerate(dataset):
        batch = {'input_ids': i['input_ids'].unsqueeze(0).to('cuda'), 'visual_embeds': i['visual_embeds'].unsqueeze(0).to('cuda'),
            'attention_mask': i['attention_mask'].unsqueeze(0).to('cuda'),
        'token_type_ids': i['token_type_ids'].unsqueeze(0).to('cuda'), 
        'visual_attention_mask': i['visual_attention_mask'].unsqueeze(0).to('cuda'), 'visual_token_type_ids': i['visual_token_type_ids'].unsqueeze(0).to('cuda')
        }

        # batch = {'input_ids': i['input_ids'].unsqueeze(0).to('cuda'), 'visual_embeds': i['visual_embeds'].unsqueeze(0).to('cuda')}

        # print(batch)

        index_to_gather = batch['attention_mask'].sum(1) - 2  # as in original code

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
        true_lab = i['labels'].argmax().item()

        
        # print(pooled_output_.shape)
        # print(pred_lab, true_lab)
        all_data_y.append(pooled_output_.squeeze(dim=1).cpu())
            
        # print(tokenizer.batch_decode(i['input_ids'].unsqueeze(0), skip_special_tokens=True))
        # print(config.id2label[pred_lab], config.id2label[true_lab])
        # print('*'*10)

        if p==args.num_examples-1:
            break

    
     








        

import numpy as np

# Define the neural network architecture
class MappingNetwork(nn.Module):
    def __init__(self, input_size=512, output_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

A_tensor = torch.from_numpy(np.asarray(all_data)).squeeze()
all_data_y = np.asarray([i for i in all_data_y])
B_tensor = torch.from_numpy(np.asarray(all_data_y)).squeeze()

print(A_tensor.shape, B_tensor.shape)

# Create dataset and dataloader
class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    
    def __len__(self):
        return len(self.A)
    
    def __getitem__(self, idx):
        return self.A[idx], self.B[idx]

dataset_mapped = VectorDataset(A_tensor, B_tensor)
dataloader = torch.utils.data.DataLoader(dataset_mapped, batch_size=50, shuffle=True)

# Initialize the model, loss function, and optimizer
model_T = MappingNetwork().to('cuda')
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model_T.parameters(), lr=0.0001)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    total_loss = []
    for A_batch, B_batch in dataloader:
        # Forward pass
        outputs = model_T(A_batch.to('cuda'))
        loss = loss_function(outputs, B_batch.to('cuda'))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
    
    # Print progress every 100 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {np.mean(np.asarray(total_loss))}')



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

model_T.eval()


import json
with open('./concept_store_vqa_temp.json', 'r') as file:
    data = json.load(file)
vocabulary = list(set(data.values()))

BATCH_SIZE = 50


# torch.use_deterministic_algorithms(False)




def get_overall(model, model_T, clip_model, clip_model_processor, image_path, batch):

    dd = {}


    image = Image.open(image_path)

  
    if args.model.lower()=='lxmert':
    
        pred_lab, output = model(batch)
        output = output.squeeze(dim=1)
        # print('ops shape', output.shape)
        pooled_output = model.dropout(output)
        logits = model.cls(pooled_output)
        # print('logits shape', logits.shape)

        
        idx = pred_lab[0].item()

    
        output.retain_grad()
        logits[:,idx].backward()
        
        grad = output.grad



    elif args.model.lower()=='visualbert':
        index_to_gather = batch['attention_mask'].sum(1) - 2  # as in original code

      
        output = model.visual_bert(
        **batch
        )

        sequence_output = output[0]

        # TO-CHECK: From the original code
        index_to_gather = (
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        )
        output = torch.gather(sequence_output, 1, index_to_gather)
        output = output.squeeze(dim=1)
        # print('ops shape', output.shape)
        pooled_output = model.dropout(output)

        logits = model.cls(pooled_output)
        reshaped_logits = logits.view(-1, model.num_labels)
        # print('logits shape', reshaped_logits.shape)
        idx = reshaped_logits.argmax(dim=-1).item()
        output.retain_grad()
        logits[:,idx].backward()
        
        grad = output.grad
       


    # print(grad, grad.shape)
    # Define the range
    lower_bound = 0.85
    upper_bound = 1.0
    
    # Calculate the mean and standard deviation for the normal distribution
    mean = (lower_bound + upper_bound) / 2  # Center of the range
    std_dev = (upper_bound - lower_bound) / 6  # Approximation to fit 99.7% of values within the range
    
    # Generate a 1x768 vector from the normal distribution
    vector = np.random.normal(mean, std_dev, 512)
    
    # Clip the values to ensure they lie between 0.85 and 1.0
    vector_clipped = torch.tensor(np.clip(vector, 0.85, 1.0)).float().to('cuda')

    img_inputs = clip_processor(images=image, return_tensors="pt")['pixel_values'].to('cuda')

    # print(img_inputs)

    image_features = clip_model.get_image_features(pixel_values = img_inputs)

    loops = int(np.ceil(len(vocabulary)/BATCH_SIZE))
    
    # for i in tqdm(range(loops)):
    for i in range(loops):
        # txt = list(model.config.id2label.values())[i*50:(i+1)*50]
        
        txt = list(vocabulary)[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        try:
        
            inputs = tokenizer(txt, padding=True, return_tensors="pt")

        except:
            break
        
        text_features = clip_model.get_text_features(input_ids = inputs['input_ids'].to('cuda'), attention_mask = inputs['attention_mask'].to('cuda'))

        
        
        
        # text_features = vector_clipped * text_features + text_features

        text_features =  text_features + text_features
        
        epsilon = 0.0000000001
        
        enc = model_T(text_features)

        # saliency = (model.classifier(outputs + epsilon * enc)[:,idx] - model.classifier(outputs)[:,idx]) / epsilon

        # saliency = list(saliency.detach().cpu().numpy())

        
        saliency = torch.matmul(grad, enc.T).squeeze()
        saliency = list(saliency.detach().cpu().numpy())
        
        for i,j in zip(txt,saliency):
            # taking absolute saliency because high negative scores also mean that "enc" influences the prediction
            # only saliencies close to 0 depicts meaningless concepts
            dd[i] = abs(j)
        
        
    top_20_pairs = sorted(dd.items(), key=lambda item: item[1], reverse = True)[:50]
    bottom_20_pairs = sorted(dd.items(), key=lambda item: item[1])[:50]
    top_20 = list(map(lambda x: x[0], top_20_pairs))
    bottom_20 = list(map(lambda x: x[0], bottom_20_pairs))

    # print(top_20)
    # print(bottom_20)
    
    inputs_top = tokenizer(top_20, padding=True, return_tensors="pt")
    inputs_bottom = tokenizer(bottom_20, padding=True, return_tensors="pt")
    

    top_text_features = clip_model.get_text_features(input_ids = inputs_top['input_ids'].to('cuda'), 
                                                     attention_mask = inputs_top['attention_mask'].to('cuda'))

    bottom_text_features = clip_model.get_text_features(input_ids = inputs_bottom['input_ids'].to('cuda'), 
                                                     attention_mask = inputs_bottom['attention_mask'].to('cuda'))

    with torch.no_grad():
        tf = torch.matmul(top_text_features, image_features.to('cuda').T).squeeze()
        bf = torch.matmul(bottom_text_features, image_features.to('cuda').T).squeeze()

    tf = tf.detach().cpu().numpy()
    bf = bf.detach().cpu().numpy()
    # print(tf, bf)

    tf_agmx = list(reversed(tf.argsort()))[:10]
    bf_agmx = list(reversed(bf.argsort()))[:10]

    top10_sorted_by_clip = []
    bottom10_sorted_by_clip = []
    
    for i,j in zip(tf_agmx, bf_agmx):
        # print(top_20_pairs[i], "  ", bottom_20_pairs[j])
        top10_sorted_by_clip.append(top_20_pairs[i][0])
        bottom10_sorted_by_clip.append(bottom_20_pairs[j][0])
        
    # print(top10_sorted_by_clip, bottom10_sorted_by_clip)
    # print('*'*10)
    inputs_top = tokenizer(top10_sorted_by_clip, padding=True, return_tensors="pt")
    inputs_bottom = tokenizer(bottom10_sorted_by_clip, padding=True, return_tensors="pt")
    

    top_text_features = clip_model.get_text_features(input_ids = inputs_top['input_ids'].to('cuda'), 
                                                     attention_mask = inputs_top['attention_mask'].to('cuda'))

    bottom_text_features = clip_model.get_text_features(input_ids = inputs_bottom['input_ids'].to('cuda'), 
                                                     attention_mask = inputs_bottom['attention_mask'].to('cuda'))

    # print(top_text_features.shape, bottom_text_features.shape)
    positive_text_features = top_text_features.mean(dim=0)
    negative_text_features = bottom_text_features.mean(dim=0)
    return dd, positive_text_features.detach().cpu(), negative_text_features.detach().cpu(), output.detach().cpu(), top10_sorted_by_clip, bottom10_sorted_by_clip 

import random


random.seed(42)
import numpy as np
np.random.seed(42)
torch.manual_seed(42)

anchors, negatives, positives = [],[],[]
T,B = [],[]

for p,i in tqdm(enumerate(dataset)):


    if args.model.lower()=='lxmert':
        batch = {'input_ids': i['input_ids'].unsqueeze(0).to('cuda'), 'visual_feats': i['visual_feats'].unsqueeze(0).to('cuda'), 
        'visual_pos': i['visual_pos'].unsqueeze(0).to('cuda'), 'attention_mask': i['attention_mask'].unsqueeze(0).to('cuda'),
        'token_type_ids': i['token_type_ids'].unsqueeze(0).to('cuda')}

        try:
            dd, pos, neg, anchor, top, bottom = get_overall(model, model_T, clip_model, clip_processor, image_path, batch)
        except:
            continue
        
    
    elif args.model.lower()=='visualbert':
        batch = {'input_ids': i['input_ids'].unsqueeze(0).to('cuda'), 'visual_embeds': i['visual_embeds'].unsqueeze(0).to('cuda'),
            'attention_mask': i['attention_mask'].unsqueeze(0).to('cuda'),
        'token_type_ids': i['token_type_ids'].unsqueeze(0).to('cuda'), 
        'visual_attention_mask': i['visual_attention_mask'].unsqueeze(0).to('cuda'), 'visual_token_type_ids': i['visual_token_type_ids'].unsqueeze(0).to('cuda')
        }
        try:
            dd, pos, neg, anchor, top, bottom = get_overall(model, model_T, clip_model, clip_processor, image_path, batch)
        except:
            continue
        

   
    anchors.append(anchor)
    negatives.append(neg)
    positives.append(pos)


    if p==args.num_examples-1:
        break

positives_ =  torch.stack(positives)
negatives_ = torch.stack(negatives)
anchors_ = torch.stack(anchors).squeeze()

print(positives_.shape, negatives_.shape, anchors_.shape)

torch.save(positives_, "./{}_{}_{}_positive.pt".format(args.model, args.dataset, str(args.num_examples)))
torch.save(negatives_, "./{}_{}_{}_negative.pt".format(args.model, args.dataset, str(args.num_examples)))
torch.save(anchors_, "./{}_{}_{}_anchor.pt".format(args.model, args.dataset, str(args.num_examples)))
