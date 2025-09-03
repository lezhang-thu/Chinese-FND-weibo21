import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys
import random
import pickle as pkl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

MAX_LEN = 256
BATCH_SIZE = 32
SEED = int(sys.argv[1])
print('SEED: {}'.format(SEED))
# URL = 'bert-base-chinese'
URL = '/home/ubuntu/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f'


# Custom dataset class for news data
class NewsDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data.iloc[idx]['content']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


# Set up the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(URL)

# Load and split the dataset (70% train, 15% val, 15% test)
data = pd.read_csv('mcfend/news.csv')
data = data[data['platform'] == '微博']
data['label'] = data['label'].map({'谣言': 1, '事实': 0})
print(data)
print('#' * 20)
train_data, temp_data = train_test_split(data,
                                         test_size=0.3,
                                         random_state=SEED)
val_data, test_data = train_test_split(temp_data,
                                       test_size=0.5,
                                       random_state=SEED)

# Create datasets
train_dataset = NewsDataset(train_data, tokenizer, max_length=MAX_LEN)
val_dataset = NewsDataset(val_data, tokenizer, max_length=MAX_LEN)
test_dataset = NewsDataset(test_data, tokenizer, max_length=MAX_LEN)


# Define a custom collate class to move batches to GPU
class GPUCollate:

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        # Assuming each sample is a dict with 'input_ids', 'attention_mask', and 'label'
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack(
            [item['attention_mask'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])

        # Move tensors to the specified device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }


# Create DataLoaders
collate_fn = GPUCollate('cuda')
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          collate_fn=collate_fn,
                          shuffle=True)
val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        collate_fn=collate_fn,
                        shuffle=False)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         collate_fn=collate_fn,
                         shuffle=False)


# Define the BERT-based classifier
class BertClassifier(nn.Module):

    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(URL)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        pooled_output = outputs.hidden_states[-1][:, 0, :]
        logits = self.linear(pooled_output)
        return logits


# Instantiate the model
model = BertClassifier().cuda()

# Set up the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-1)
loss_fn = nn.BCEWithLogitsLoss()

# Training loop with early stopping based on macro F1 score
num_epochs = 20
# patience = 3
patience = num_epochs
best_val_f1 = 0.0
epochs_without_improvement = 0

total_training_steps = num_epochs * len(train_loader)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.06 * total_training_steps),
    num_training_steps=total_training_steps)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits.view(-1), labels)
        loss.backward()
        optimizer.step()

        scheduler.step()

        if random.uniform(0, 1) < 1e-1:
            print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

    # Validation: accumulate predictions and labels
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).view(-1)
            preds = (probs > 0.5).float()

            val_preds.extend(preds.cpu().numpy().astype(int))
            val_labels.extend(labels.cpu().numpy().astype(int))

    # Compute macro F1 score for validation
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Validation Macro F1: {val_f1:.4f}")

    # Early stopping check
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).view(-1)
        preds = (probs > 0.5).float()

        test_preds.extend(preds.cpu().numpy().astype(int))
        test_labels.extend(labels.cpu().numpy().astype(int))

# Compute macro F1 score for test set
test_f1 = f1_score(test_labels, test_preds, average='macro')
test_acc = accuracy_score(test_labels, test_preds)
with open('optimized-bert-ft-s{}.pkl'.format(SEED), 'wb') as f:
    pkl.dump({
        "y_true": test_labels,
        "y_pred": test_preds,
    }, f)
print(f"Test Macro F1: {test_f1:.4f}")
print(f"Test Acc: {test_acc:.4f}")
