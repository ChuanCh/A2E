import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, outputs, targets):
        # Normalize outputs and targets to unit vectors
        outputs_norm = F.normalize(outputs, p=2, dim=1)
        targets_norm = F.normalize(targets, p=2, dim=1)
        # Compute cosine similarity
        cosine_loss = 1 - torch.sum(outputs_norm * targets_norm, dim=1).mean()
        return cosine_loss

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path  # Save path for the best model

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.val_loss_min > val_loss:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss

def train_model(model, dataloader, val_dataloader, starting_epoch, device, criterion, optimizer, scheduler, writer, early_stopper, chkpt_dir, num_epochs=100):
    """
    Train a model with the given data, optimizer, loss function, and other parameters.

    Args:
    - model (torch.nn.Module): The model to train.
    - dataloader (DataLoader): The DataLoader for training data.
    - val_dataloader (DataLoader): The DataLoader for validation data.
    - device (torch.device): The device to run the training on (CPU or GPU).
    - criterion (function): The loss function.
    - optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
    - scheduler (torch.optim.lr_scheduler): Scheduler for learning rate adjustment.
    - writer (SummaryWriter): TensorBoard writer for logging.
    - early_stopper (EarlyStopping): Instance of EarlyStopping to control the training process based on validation loss.
    - chkpt_dir (str): Directory to save checkpoints.
    - best_model_path (str): Path to save the best model.
    - num_epochs (int): Number of epochs to train the model.

    Returns:
    - None
    """
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        # Initialize the progress bar for the current epoch
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{num_epochs}", leave=False, mininterval=10)


        for i, data in progress_bar:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Update the progress bar description
            progress_bar.set_postfix({'Training Loss': loss.item()})

            if i % 1000 == 0:  # Log every 1000 batches
                writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + i)

        average_loss = running_loss / len(dataloader)
        writer.add_scalar('Average Training Loss', average_loss, epoch + 1)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for data in val_dataloader:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_dataloader)
        writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        print(f"Validation Loss: {val_loss}")

        

        # Early stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping")
            break

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        # Save checkpoint
        if epoch % 1 == 0:  # Save every 1 epoch in chkpt folder
            checkpoint_path = os.path.join(chkpt_dir, f'checkpoint_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': running_loss / len(dataloader),
                'val_loss': val_loss,
                'lr': current_lr
            }, checkpoint_path)
            if early_stopper.verbose:
                print(f"Early stop: Checkpoint saved at {checkpoint_path}")

        for name, param in model.named_parameters():
            writer.add_histogram(f'{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, epoch)

def load_checkpoint(model, optimizer, scheduler, chkpt_dir, start_epoch=0):
    """
    Load a checkpoint if exists and return the epoch number to resume training from.

    Args:
    - model (torch.nn.Module): The model to load state dict into.
    - optimizer (torch.optim.Optimizer): The optimizer to load state dict into.
    - scheduler (torch.optim.lr_scheduler): Scheduler to load the state dict into.
    - chkpt_dir (str): Directory where checkpoints are saved.
    - start_epoch (int, optional): The starting epoch if no checkpoint is found.

    Returns:
    - model (torch.nn.Module): The model with loaded state dict.
    - optimizer (torch.optim.Optimizer): The optimizer with loaded state dict.
    - scheduler (torch.optim.lr_scheduler): The scheduler with loaded state dict.
    - start_epoch (int): The epoch training should resume from.
    """
    import os
    latest_chkpt_path = None
    highest_epoch = -1

    # Search for the latest checkpoint
    for filename in os.listdir(chkpt_dir):
        if filename.startswith('checkpoint_') and filename.endswith('.pt'):
            epoch_num = int(filename.split('_')[1].split('.')[0])
            if epoch_num > highest_epoch:
                highest_epoch = epoch_num
                latest_chkpt_path = os.path.join(chkpt_dir, filename)

    # Load the latest checkpoint if exists
    if latest_chkpt_path is not None:
        checkpoint = torch.load(latest_chkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Loaded checkpoint '{latest_chkpt_path}' (epoch {checkpoint['epoch']}).")
    else:
        print("No checkpoint found. Starting a new model.")
        best_val_loss = float('inf')


    return model, optimizer, scheduler, start_epoch, best_val_loss
