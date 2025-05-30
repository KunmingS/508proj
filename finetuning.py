import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from llama.model import Llama, ModelArgs
from llama.tokenizer import Tokenizer
from torch import nn, optim
from tqdm import tqdm
import requests

#parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
MODEL_PATH = os.path.join(checkppoint_dir, "tokenizer.model")
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 5e-3
BATCH_SIZE = 1
EPOCHS = 3
USE_MIXED_PRECISION = True
USE_LORA = True
USE_CHECKPOINT = False

#define the Alpaca dataset
class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]["instruction"]
        if self.data[idx]["input"]:
            prompt += "\n" + self.data[idx]["input"]
        target = self.data[idx]["output"]

        combined_text = prompt + target
        
        combined_ids = self.tokenizer.encode(combined_text, bos=True, eos=True)
        prompt_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        
        labels = [-100] * len(prompt_ids) + combined_ids[len(prompt_ids):]

        return torch.tensor(combined_ids), torch.tensor(labels)
    
    def get_first_data_item(self):
        prompt = self.data[0]["instruction"]
        if self.data[0]["input"]:
            prompt += "\n" + self.data[0]["input"]
        print(prompt)
        prompt_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        return torch.tensor(prompt_ids).unsqueeze(0)
        

#pading aligning
def collate(batch):
    input_ids, labels = zip(*batch)
    max_len = max(len(x) for x in input_ids)
    input_ids = [torch.cat([x, torch.full((max_len - len(x),), 0)]) for x in input_ids]
    labels = [torch.cat([y, torch.full((max_len - len(y),), -100)]) for y in labels]
    return torch.stack(input_ids), torch.stack(labels)

def test_model(model, dataset, tokenizer):
    model.eval()
    first_data_item = dataset.get_first_data_item()
    res = model(first_data_item.to(DEVICE), start_pos=0)
    # Get the token IDs from the model's output
    token_ids = res.argmax(dim=-1)[0].tolist()
    decoded_output = tokenizer.decode(token_ids)
    print(decoded_output)

#load tokenizer and data
tokenizer = Tokenizer(MODEL_PATH)
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
response = requests.get(url)
data = response.json()
data = data[:200]

dataset = AlpacaDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

args = ModelArgs()
args.kv_caching = False
args.use_checkpoint = USE_CHECKPOINT
model = Llama(args)

model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint, strict=False)  
model = model.to(DEVICE)

for name, param in model.named_parameters():
    if USE_LORA:
        if not any(x in name for x in ['lora', 'A', 'B']):
            param.requires_grad = False
        else:
            print(f"Trainable parameter: {name}")
    else:
        param.requires_grad = True
        print(f"Trainable parameter: {name}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params}, Total: {total_params}, Ratio: {trainable_params/total_params:.4f}")

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler() if USE_MIXED_PRECISION else None
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

test_model(model, dataset, tokenizer)

model.train()
step = 0
step_times = []
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    total_loss = 0
    for i, (input_ids, labels) in enumerate(tqdm(dataloader)):
        start = time.time()
        input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            
        if USE_MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, start_pos=0)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
        else:
            outputs = model(input_ids, start_pos=0)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss = loss / GRAD_ACCUM_STEPS
            
        total_loss += loss.item()
        
        if USE_MIXED_PRECISION:
            scaler.scale(loss).backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            if USE_MIXED_PRECISION:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            print(f"Step {i + 1}, Loss: {total_loss:.4f}")
            with open("training_log.txt", "a") as f:
                f.write(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {total_loss:.4f}\n")

            total_loss = 0
            
        end = time.time()
        step_times.append(end - start)

print(f"Lora: {USE_LORA}, Mixed Precision: {USE_MIXED_PRECISION}, Checkpoint: {USE_CHECKPOINT}")
avg_step_time = sum(step_times) / len(step_times)
print(f"Average step time: {avg_step_time:.2f}s")
print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f}MB")
# print status of lora and mixed precision


