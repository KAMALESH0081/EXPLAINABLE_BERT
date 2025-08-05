import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from tqdm import tqdm
import os
import warnings

epochs = 10
lr = 10**-4
batch_size = 32

train_dataloader = "dummy_train_dataloader"  # Placeholder for actual DataLoader

def train_model(model, load_path, save_path ):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    checkpoint_path = load_path
    initial_epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading model and optimizer state...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        print(f"Model loaded from checkpoint. Resuming from epoch {initial_epoch + 1}.")
    else:
        print("Checkpoint not found. Training from scratch.")

    for epoch in range(initial_epoch, epochs):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0  # To track loss
        i = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch+1:02d}")
        for batch in batch_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['pad_mask'].to(device)
            targets = batch['label'].to(device)

            # Forward pass
            output = model(input_ids, attention_mask)
            loss = loss_fn(output, targets)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            i += 1
        # Save model checkpoint with epoch number and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)

        print(f"Checkpoint saved at epoch {epoch + 1}.")
        #run_validation(model, custom_tokenizer, 16, device, lambda msg: batch_iterator.write(msg))