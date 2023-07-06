import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

# Define a class ASClassificationDataset that inherits from PyTorch's Dataset class
class ASClassificationDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, idx: int):
        return self.features[idx, :], self.labels[idx]

# Define a class ASClassifier that inherits from PyTorch's Module class
class ASClassifier(nn.Module):
    # The __init__ method initializes the class instances
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Define a sequential model stored in 'self.net'
        self.net = nn.Sequential(
            # First linear layer with 'input_dim' inputs and approximately 2/3 of the sum of 'input_dim' and 'output_dim' outputs
            nn.Linear(input_dim, round((input_dim + output_dim) * .66)),
            # Apply a ReLU activation function
            nn.ReLU(),
            # Second linear layer with approximately 2/3 of the sum of 'input_dim' and 'output_dim' inputs and outputs
            nn.Linear(round((input_dim + output_dim) * .66), round((input_dim + output_dim) * .66)),
            # Apply another ReLU activation function
            nn.ReLU(),
            # Third linear layer with approximately 2/3 of the sum of 'input_dim' and 'output_dim' inputs and 'output_dim' outputs
            nn.Linear(round((input_dim + output_dim) * .66), output_dim)
        )

    # The forward method defines the forward pass of the model
    def forward(self, X):
        # Pass the input 'X' through 'self.net' to compute the output, and store it in 'logits'
        logits = self.net(X)
        return logits

def main(data):
    #Declare random seeds for the testing
    torch.manual_seed(2)
    np.random.seed(2)
    # load data from files
    train_features = np.loadtxt(f"{data}instance-features.txt")
    train_performance = np.loadtxt(f"{data}performance-data.txt")
    print(train_features.shape)
    print(train_performance.shape)

    # Normalize the features using min-max normalization
    min_features = np.min(train_features, axis=0)
    max_features = np.max(train_features, axis=0)
    eps = 1e-20
    train_features = (train_features - min_features) / (max_features - min_features + eps)

    train_labels = np.argmin(train_performance, axis=1)

    print(train_labels[0])

    # convert data to the types required by pytorch
    train_features = train_features.astype(np.float32)
    # Create the dataset and split it into training and validation sets
    dataset = ASClassificationDataset(train_features, train_labels)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for the training and validation datasets. Shuffle the training data.
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model with the number of features and unique labels in the training data.
    model = ASClassifier(train_features.shape[1], len(np.unique(train_labels)))

    # Define the loss function as cross-entropy loss.
    loss_function = nn.CrossEntropyLoss(reduction="sum")

    # Define the optimizer as Adam with learning rate 0.01 and weight decay 0.001.
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.001)

    # Define the maximum number of epochs, number of batches, number of samples, patience for early stopping, 
    # and tolerance for early stopping.
    max_epochs = 1000
    n_batches = len(train_dataloader)
    n_samples = train_features.shape[0]
    patience = 40
    tolerance = 100
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    # Train the model for max_epochs epochs.
    for ep_id in range(max_epochs):
        total_loss = 0
        total_n_corrects = 0

        # For each batch in the training data.
        for X, y in train_dataloader:
            # Make a forward pass through the model.
            pred = model(X)

            # Compute the loss.
            loss = loss_function(pred, y)

            # Compute the number of correct predictions.
            pred_best_algo = torch.argmax(pred, dim=1)
            true_best_algo = y
            n_corrects = (pred_best_algo == true_best_algo).sum()

            # Zero the gradients, perform a backward pass, and update the weights.
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss
            total_n_corrects += n_corrects

        # Compute the average loss and accuracy for this epoch.
        acc = total_n_corrects / n_samples
        avg_loss = total_loss / n_batches
        print(f"epoch: {ep_id:4d},\t loss: {avg_loss: 5.4f},\t accuracy: {acc: 3.4f}")

        # Evaluate the model on the validation data.
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                pred_val = model(X_val)
                val_loss = loss_function(pred_val, y_val)
                total_val_loss += val_loss.item()

        # Compute the average validation loss for this epoch.
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss: 5.4f}")

        # Check for early stopping: if the validation loss has not improved for 'patience' epochs and we have passed 'tolerance' epochs, stop the training.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience and ep_id > tolerance:
                print(f"Early stopping at epoch {ep_id}.")
                break

        # Save the training and validation losses for this epoch.
        train_losses.append(avg_loss.detach().numpy())
        val_losses.append(avg_val_loss)

    return model