from data_preprocessing import *
from transformer import *
from datasets import load_dataset
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss



""" collecting training data """

print("Collecting data...")
train_dataset = load_dataset("kmfoda/booksum")
train_raw_data = train_dataset['train'].to_pandas()
train_columns = train_raw_data[['chapter', 'summary_text']]
train_data = pd.DataFrame(train_columns)
train_data.drop_duplicates(subset=['chapter'], inplace=True)
print("Done: collected training data")


""" collecting testing data """

test_dataset = load_dataset("kmfoda/booksum")
test_raw_data = test_dataset['test'].to_pandas()
test_columns = test_raw_data[['chapter', 'summary_text']]
test_data = pd.DataFrame(test_columns)
test_data.drop_duplicates(subset=['chapter'], inplace=True)
print("Done: collected test data")


""" cleaning training Data """

print("Cleaning Data...")
train_data['clean_chapter'] = train_data['chapter'].apply(text_cleaner)
train_data['clean_summary_text'] = train_data['summary_text'].apply(text_cleaner)
train_data['clean_summary_text'] = train_data['clean_summary_text'].replace('', np.nan)
train_data.dropna(axis=0, inplace=True)
print("Done: cleaned data")

# specifying maximum length
max_len_chapter = 20000
max_len_summary = 2500

# create tokens
input_tokens = tokenize(train_data['clean_chapter'])
output_tokens = tokenize(train_data['clean_summary_text'])

print("Done: tokens created")

# create vocabulary
input_vocab = Vocab(input_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>'])
output_vocab = Vocab(output_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>'])

print("Done: vocabulary created")

input_array, src_valid_len = build_array_sum(input_tokens, input_vocab, max_len_chapter)
output_array, tgt_valid_len = build_array_sum(output_tokens, output_vocab, max_len_summary)
data_arrays = (input_array, src_valid_len, output_array, tgt_valid_len)

batch_size = 64
data_iter = load_array(data_arrays, batch_size)


""" Initialize model """
model = Transformer(
    src_vocab_size=len(input_vocab),
    tgt_vocab_size=len(output_vocab),
    d_model=512,        
    num_heads=8,        
    num_layers=6,       
    d_ff=2048,          
    max_seq_length=max_len_chapter,
    dropout=0.1
)


""" Define loss function and optimizer """
criterion = CrossEntropyLoss(ignore_index=input_vocab['<pad>'])  
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)


""" Training Loop """
def train_epoch(data_iter, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for src, src_valid_len, tgt, tgt_valid_len in data_iter:
        optimizer.zero_grad()
        
        output = model(src, tgt[:, :-1])  # Pass input and target shifted by one for teacher forcing
        tgt_y = tgt[:, 1:]  # Target labels are offset by one position
        
        # Compute loss
        loss = criterion(output.view(-1, output.shape[-1]), tgt_y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_iter)


""" Training Loop over Epochs """
print("Training Loop started...\n")
EPOCHS = 50
for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1} started...')
    loss = train_epoch(data_iter, model, criterion, optimizer)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}\n')
