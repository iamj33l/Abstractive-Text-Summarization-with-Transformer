from data_preprocessing import *
from transformer import *
from datasets import load_dataset
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import numpy as np

""" Collecting training data """
print("Collecting data...")
train_dataset = load_dataset("cnn_dailymail", "3.0.0")
train_raw_data = train_dataset['train'].to_pandas()
train_columns = train_raw_data[['article', 'highlights']]
train_data = pd.DataFrame(train_columns)
train_data.drop_duplicates(subset=['article'], inplace=True)
print("Done: collected training data")

""" Collecting testing data """
test_dataset = load_dataset("cnn_dailymail", "3.0.0")
test_raw_data = test_dataset['test'].to_pandas()
test_columns = test_raw_data[['article', 'highlights']]
test_data = pd.DataFrame(test_columns)
test_data.drop_duplicates(subset=['article'], inplace=True)
print("Done: collected test data")

""" Cleaning training Data """
print("Cleaning training Data...")
train_data['clean_article'] = train_data['article'].apply(text_cleaner)
train_data['clean_highlights'] = train_data['highlights'].apply(text_cleaner)
train_data['clean_highlights'] = train_data['clean_highlights'].replace('', np.nan)
train_data.dropna(axis=0, inplace=True)
print("Done: cleaned training data")

""" Cleaning testing Data """
print("Cleaning testing Data...")
test_data['clean_article'] = test_data['article'].apply(text_cleaner)
test_data['clean_highlights'] = test_data['highlights'].apply(text_cleaner)
test_data['clean_highlights'] = test_data['clean_highlights'].replace('', np.nan)
test_data.dropna(axis=0, inplace=True)
print("Done: cleaned testing data")

# Specifying maximum length
max_len_chapter = 1250  # Reduced from 20000 to fit memory constraints
max_len_summary = 150   # Reduced from 2500 to fit memory constraints

# Create tokens
input_tokens = tokenize(train_data['clean_article'])
output_tokens = tokenize(train_data['clean_highlights'])
print("Done: tokens created")

# Create vocabulary
input_vocab = Vocab(input_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>'])
output_vocab = Vocab(output_tokens, reserved_tokens=['<pad>', '<bos>', '<eos>'])
print("Done: vocabulary created")

# Prepare arrays for input and output
input_array, src_valid_len = build_array_sum(input_tokens, input_vocab, max_len_chapter)
output_array, tgt_valid_len = build_array_sum(output_tokens, output_vocab, max_len_summary)
data_arrays = (input_array, src_valid_len, output_array, tgt_valid_len)

# Set batch size (keeping it small for memory reasons)
batch_size = 1
data_iter = load_array(data_arrays, batch_size)

""" Initialize model """
model = Transformer(
    src_vocab_size=len(input_vocab),
    tgt_vocab_size=len(output_vocab),
    d_model=256,        
    num_heads=4,  
    num_layers=4,       
    d_ff=512,      
    max_seq_length=max_len_chapter,
    dropout=0.1
)

# Move model to device (CPU in your case)
device = torch.device("cpu")
model.to(device)

""" Define loss function and optimizer """
criterion = CrossEntropyLoss(ignore_index=input_vocab['<pad>'])  # Loss function ignores padding tokens
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# Gradient accumulation to simulate larger batch size
accumulation_steps = 4  # Accumulate gradients over 4 steps

""" Training Loop """
def train_epoch(data_iter, model, criterion, optimizer, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (src, src_valid_len, tgt, tgt_valid_len) in enumerate(data_iter):
        # Move data to the same device as the model
        src, tgt = src.to(device), tgt.to(device)

        # Forward pass
        output = model(src, tgt[:, :-1])  # Target shifted by one for teacher forcing
        tgt_y = tgt[:, 1:]  # Target labels are offset by one position
        loss = criterion(output.view(-1, output.shape[-1]), tgt_y.reshape(-1))
        loss = loss / accumulation_steps  # Normalize the loss for accumulated gradients

        # Backward pass
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # Perform optimizer step
            optimizer.zero_grad()  # Zero out the gradients

        total_loss += loss.item()

    return total_loss / len(data_iter)

""" Training Loop over Epochs """
print("Training Loop started...\n")
EPOCHS = 50
for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1} started...')
    loss = train_epoch(data_iter, model, criterion, optimizer, accumulation_steps)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}\n')

print("Training Completed!")

""" Tesing the model """

# Prepare arrays for input and output
input_tokens_test = tokenize(test_data['clean_article'])
output_tokens_test = tokenize(test_data['clean_highlights'])

input_array_test, src_valid_len_test = build_array_sum(input_tokens_test, input_vocab, max_len_chapter)

# Set batch size (keeping it small for memory reasons)
batch_size_test = 1
data_iter_test = load_array((input_array_test, src_valid_len_test), batch_size_test)

def predict(data_iter, model, output_vocab, max_len_summary):
    model.eval()
    results = []

    for i, (src, src_valid_len) in enumerate(data_iter):
        # Move data to the same device as the model
        src = src.to(device)

        # Initialize output with bos token
        tgt = torch.tensor([[output_vocab['<bos>']]]).to(device)

        for i in range(max_len_summary):
            # Forward pass
            output = model(src, tgt)
            pred = output.argmax(dim=-1)[:, -1].unsqueeze(1)

            # Concatenate the prediction to the output
            tgt = torch.cat((tgt, pred), dim=-1)

            # Stop if the model predicts the eos token
            if pred[0][0].item() == output_vocab['<eos>']:
                break

        results.append(tgt[0].cpu().numpy())

    return results

# Predict summaries
print("Predicting summaries...")
results = predict(data_iter_test, model, output_vocab, max_len_summary)
print("Done: predicted summaries")

# Convert the predicted summaries to text
predicted_summaries = [output_vocab.to_tokens(res) for res in results]

# Convert the actual summaries to text
actual_summaries = test_data['clean_highlights'].tolist()

# Display the first 5 predicted and actual summaries
for i in range(5):
    print(f"\nPredicted Summary: {predicted_summaries[i]}")
    print(f"Actual Summary: {actual_summaries[i]}")


''' Save the model '''
print("\nSaving model...")
torch.save(model.state_dict(), 'transformer_summarization_model.pth')
print("\nModel saved!")
