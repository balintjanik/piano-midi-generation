import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
import copy

def create_dataloaders(dataset, batch_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
    """
    Splits a dataset into training, validation, and testing subsets and creates corresponding DataLoaders.

    Parameters:
    - dataset (Dataset) : The dataset to split (any custom dataset class compatible with PyTorch).
    - batch_size (int) : The batch size for DataLoaders.
    - train_ratio (float) : Proportion of the dataset to use for training.
    - val_ratio (float) : Proportion of the dataset to use for validation.
    - test_ratio (float) : Proportion of the dataset to use for testing.
    - shuffle (boolean) : Whether to shuffle the dataset before splitting.

    Returns:
    - A tuple of DataLoaders: (train_loader, val_loader, test_loader)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    if shuffle:
        dataset = Subset(dataset, indices=np.random.permutation(total_size))

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def plot_metrics(train_losses, val_losses, train_accuracy, val_accuracy):
    """
    Plots training and validation metrics.
    Args:
        train_losses (list) : List of training losses
        val_losses (list) : List of validation losses
        train_accuracy (list) : List of training accuracies
        val_accuracy (list) : List of validation accuracies
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def compute_accuracy(y_hat, y, task_type):
    """
    Computes accuracy for multiclass or multi-label classification.
    Args:
        y_hat : Model predictions (logits or probabilities)
        y : Ground truth labels
        task_type (str) : 'multiclass' or 'multi-label'
    Returns:
        Accuracy as a float
    """
    if task_type == "multiclass":
        y_pred = torch.argmax(y_hat, dim=1)
        return (y_pred == y).float().mean().item()
    elif task_type == "multi-label":
        y_pred = (torch.sigmoid(y_hat) > 0.5).float()
        return (y_pred == y).float().mean().item()
    else:
        raise ValueError("Unsupported task_type. Use 'multiclass' or 'multi-label'.")

def train_one_epoch(model, dataloader, optimizer, loss_fn, task_type, device="cpu"):
    """
    Trains the model for one epoch.
    Args:
        model : The PyTorch model to train
        dataloader : DataLoader for training data
        optimizer : Optimizer for the model
        loss_fn : Loss function
        task_type (str) : 'multiclass' or 'multi-label'
        device (str) : Device ('cpu' or 'cuda')
    Returns:
        Tuple of average loss and accuracy for the epoch
    """
    model.train()
    epoch_losses = []
    epoch_accs = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        acc = compute_accuracy(y_hat, y, task_type)
        epoch_losses.append(loss.item())
        epoch_accs.append(acc)
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accs)
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate_one_epoch(model, dataloader, loss_fn, task_type, device="cpu"):
    """
    Evaluates the model for one epoch.
    Args:
        model : The PyTorch model to evaluate
        dataloader : DataLoader for validation or test data
        loss_fn : Loss function
        task_type (str) : 'multiclass' or 'multi-label'
        device (str) : Device ('cpu' or 'cuda')
    Returns:
        Tuple of average loss and accuracy for the epoch
    """
    model.eval()
    epoch_losses = []
    epoch_accs = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        acc = compute_accuracy(y_hat, y, task_type)
        epoch_losses.append(loss.item())
        epoch_accs.append(acc)

    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accs)
    return avg_loss, avg_acc

def train_model(
    model,
    dataloaders,
    task_type,
    loss_fn=None,
    optimizer=None,
    n_epochs=5,
    early_stopping_patience=2,
    device="cpu",
    learning_rate=0.001,
):
    """
    Trains and evaluates the model, with early stopping.
    Args:
        model : The PyTorch model to train
        dataloaders : Dictionary with keys 'train' and 'val'
        task_type : 'multiclass' or 'multi-label'
        loss_fn : Loss function (default determined by task_type)
        optimizer : Optimizer for the model
        n_epochs (int) : Number of epochs to train
        early_stopping_patience (int) : Number of epochs to wait for validation loss improvement
        device (str) : Device to train on ('cpu' or 'cuda')
        learning_rate (float) : Learning rate for the optimizer
    Returns:
        Best model (based on validation loss), and training/validation metrics
    """

    if loss_fn is None:
        if task_type == "multiclass":
            loss_fn = nn.CrossEntropyLoss()
        elif task_type == "multi-label":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unsupported task_type. Use 'multiclass' or 'multi-label'.")

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []
    best_model = None
    best_val_loss = float("inf")
    last_early_stopping_update_ep_idx = 0

    model.to(device)

    for epoch_idx in range(n_epochs):
        # Train
        avg_train_loss, avg_train_acc = train_one_epoch(
            model, dataloaders["train"], optimizer, loss_fn, task_type, device
        )
        train_losses.append(avg_train_loss)
        train_accuracy.append(avg_train_acc)

        # Validate
        avg_val_loss, avg_val_acc = evaluate_one_epoch(
            model, dataloaders["val"], loss_fn, task_type, device
        )
        val_losses.append(avg_val_loss)
        val_accuracy.append(avg_val_acc)

        print(
            f"Epoch {epoch_idx}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
            f"Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            last_early_stopping_update_ep_idx = epoch_idx
            print("    Early stopping: Best model updated.")
        
        if epoch_idx - last_early_stopping_update_ep_idx >= early_stopping_patience:
            print("Early stopping triggered. Restoring best model.")
            break

    return best_model, (train_losses, val_losses, train_accuracy, val_accuracy)

@torch.no_grad()
def test_model(model, dataloader, loss_fn, task_type, device="cpu"):
    """
    Evaluates the model on test data.
    Args:
        model : The trained PyTorch model
        dataloader : DataLoader for test data
        loss_fn : Loss function
        task_type : 'multiclass' or 'multi-label'
        device : Device ('cpu' or 'cuda')
    Returns:
        Test loss and accuracy
    """
    test_loss, test_acc = evaluate_one_epoch(model, dataloader, loss_fn, task_type, device)
    print(f"Test metrics - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    return test_loss, test_acc