# Import the required library
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

# Define a class ASRegressionDataset that inherits from PyTorch's Dataset class
class ASRegressionDataset(Dataset):
    def __init__(self, features: np.ndarray, performance: np.ndarray):
        self.features = features
        self.performance = performance

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx,:], self.performance[idx]

# Define a class ASRegressor that inherits from PyTorch's Module class
class ASRegressor(nn.Module):
    # The __init__ method initializes the class instances
    def __init__(self, input_dim, output_dim):
        super().__init__()
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

# load data from files
def main(data):
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    np.random.seed(2)

    # Load the feature and performance data from text files
    instance_features = np.loadtxt(f"{data}instance-features.txt")
    performance_data = np.loadtxt(f"{data}performance-data.txt")

    # Normalize the features using min-max normalization
    min_features = np.min(instance_features, axis=0)
    max_features = np.max(instance_features, axis=0)
    eps = 1e-20  # Small constant to prevent division by zero
    instance_features = (instance_features - min_features) / (max_features - min_features + eps)

    # Convert data to the types required by PyTorch
    instance_features = instance_features.astype(np.float32)
    performance_data = performance_data.astype(np.float32)
      
    # Create the custom Dataset
    dataset = ASRegressionDataset(instance_features, performance_data)

    # Split the dataset into training and validation sets (80% training, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for the training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = ASRegressor(instance_features.shape[1], performance_data.shape[1])
    loss_function = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    # Set maximum number of training epochs and initialize variables for early stopping
    max_epochs = 1000
    patience = 40
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for ep_id in range(max_epochs):
        # Initialize loss and accuracy for this epoch
        total_loss = 0
        total_n_corrects = 0

        # Loop over batches
        for X, y in train_dataloader:
            # Forward pass: Compute predictions and loss
            pred = model(X)
            loss = loss_function(pred, y)

            # Compute accuracy
            predicted_classes = pred.argmin(axis=1)
            actual_classes = y.argmin(axis=1)
            n_corrects = (predicted_classes == actual_classes).sum()

            # Zero gradients, perform backward pass, and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Update running loss and accuracy
            total_loss += loss
            total_n_corrects += n_corrects

        # Compute average loss and accuracy for this epoch
        acc = total_n_corrects / len(train_dataset)
        avg_loss = total_loss / len(train_dataloader)

        # Print training stats
        print(f"epoch: {ep_id:4d},\t loss: {avg_loss: 5.4f},\t accuracy: {acc: 3.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                pred_val = model(X_val)
                # Calculate the validation loss
                val_loss = loss_function(pred_val, y_val)
                total_val_loss += val_loss.item()

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss: 5.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            # If the current validation loss is the best we've seen so far, save it and reset the counter
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            # Otherwise, increment the counter
            epochs_without_improvement += 1
            # If we've gone a number of epochs without improvement, stop training
            if epochs_without_improvement >= patience and ep_id > 100:
                print(f"Early stopping at epoch {ep_id}.")
                break

    # Return the trained model
    return model