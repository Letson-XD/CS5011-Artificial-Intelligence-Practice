import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

class ASRegretDataset(Dataset):
    # Initialize the dataset with features, labels, and performance.
    def __init__(self, features: np.ndarray, labels: np.ndarray, performance: np.ndarray):
        self.features = features
        self.labels = labels
        self.performance = performance

    # Return the total number of samples in the dataset.
    def __len__(self):
        return self.features.shape[0]

    # Return the sample (features, label, performance) at the given index.
    def __getitem__(self, idx: int):
        return self.features[idx, :], self.labels[idx], self.performance[idx, :]
    
class ASClassifier(nn.Module):
    # The __init__ method initializes the class instances
    def __init__(self, input_dim, output_dim):
        # Call the __init__ method of the parent class 'nn.Module'
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

def loss_function(pred, y, performance, alpha=0.6):
    # Calculate cross-entropy loss
    cross_entropy_loss = nn.CrossEntropyLoss()(pred, y)
    
    # Find the predicted and true best algorithm indices
    pred_best_algo = torch.argmax(pred, dim=1)
    true_best_algo = y
    
    # Convert performance to a PyTorch tensor
    performance = torch.tensor(performance, dtype=torch.float32)
    
    # Calculate predicted and true costs
    pred_cost = performance[np.arange(len(pred_best_algo)), pred_best_algo]
    true_cost = performance[np.arange(len(true_best_algo)), true_best_algo]
    
    # Calculate regret and its loss
    regret = pred_cost - true_cost
    regret_loss = torch.min(regret)
    
    # Combine cross-entropy and regret losses using the alpha hyperparameter
    combined_loss = (alpha * cross_entropy_loss) + ((1 - alpha) * regret_loss)
    return combined_loss

def main(data):
    torch.manual_seed(2)
    np.random.seed(2)
    # load data from files
    train_features = np.loadtxt(f"{data}instance-features.txt", delimiter=" ")
    train_performance = np.loadtxt(f"{data}performance-data.txt", delimiter=" ")
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
    train_performance = train_performance.astype(np.float32)
    
    # Create the dataset and split it into training and validation sets
    dataset = ASRegretDataset(train_features, train_labels, train_performance)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ASClassifier(train_features.shape[1], len(np.unique(train_labels)))

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001) 

    max_epochs = 1000
    n_batches = len(train_dataloader)
    n_samples = train_features.shape[0]
    patience = 100
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    for ep_id in range(max_epochs):
        total_loss = 0
        total_n_corrects = 0

        # Loop over each batch in the training dataset
        for X, y, performance in train_dataloader:
            # Forward pass through the model to get the predictions
            pred = model(X)

            # Calculate the loss for the batch
            loss = loss_function(pred, y, performance)

            # Calculate the number of correct predictions for the batch
            pred_best_algo = torch.argmax(pred, dim=1)
            true_best_algo = y
            n_corrects = (pred_best_algo == true_best_algo).sum()

            # Update the gradients and optimizer parameters
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Add the loss and number of correct predictions for the batch to the totals for the epoch
            total_loss += loss
            total_n_corrects += n_corrects

        # Calculate the accuracy and average loss for the epoch
        acc = total_n_corrects / n_samples
        avg_loss = total_loss / n_batches

        # Print the epoch number, average loss, and accuracy
        print(f"epoch: {ep_id:4d},\t loss: {avg_loss: 5.4f},\t accuracy: {acc: 3.4f}")

        # Set the model to evaluation mode and initialize a variable to track the total validation loss
        model.eval()
        total_val_loss = 0

        # Loop over each batch in the validation dataset
        with torch.no_grad():
            for X_val, y_val, performance in val_dataloader:
                # Forward pass through the model to get the predictions
                pred_val = model(X_val)

                # Calculate the loss for the batch
                val_loss = loss_function(pred_val, y_val, performance)
                total_val_loss += val_loss.item()

        # Calculate the average validation loss for the epoch
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss: 5.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {ep_id}.")
                break

        # Record the training and validation losses for the epoch
        train_losses.append(avg_loss.detach().numpy())
        val_losses.append(avg_val_loss)

    return model