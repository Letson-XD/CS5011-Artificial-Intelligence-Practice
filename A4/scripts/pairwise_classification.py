import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# This class defines a custom PyTorch Dataset for pairwise algorithm comparisons.
class ASPairwiseDataset(Dataset):
    def __init__(self, features: np.ndarray, performance: np.ndarray, algo_pairs: list):
        # The features of the dataset instances.
        self.features = features
        # The performance data of the algorithm instances.
        self.performance = performance
        # The pairs of algorithms to be compared.
        self.algo_pairs = algo_pairs

    # This method returns the total number of samples in the dataset.
    def __len__(self):
        # Each instance is compared for each pair of algorithms, so the total number of samples is the number of instances times the number of pairs.
        return self.features.shape[0] * len(self.algo_pairs)

    # This method returns the sample (feature vector and performance difference) and its corresponding pair of algorithms, given an index.
    def __getitem__(self, idx: int):
        # The index of the instance is the integer division of the index by the number of pairs.
        instance_idx = idx // len(self.algo_pairs)
        # The index of the pair is the remainder of the index divided by the number of pairs.
        pair_idx = idx % len(self.algo_pairs)
        # Get the pair of algorithms.
        pair = self.algo_pairs[pair_idx]
        # Get the feature vector of the instance.
        feature_vector = self.features[instance_idx, :]
        # Calculate the performance difference between the two algorithms of the pair for the instance.
        performance_diff = self.performance[instance_idx, pair[0]] - self.performance[instance_idx, pair[1]]
        # Return the feature vector, the performance difference, and the pair of algorithms.
        return feature_vector, performance_diff, pair

# This class defines a neural network architecture for the pairwise algorithm comparison task.
class ASPairwiseRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        output_dim = 10
        
        # All layers except the output layer use the ReLU activation function. 
        self.net = nn.Sequential(
            # The input layer. It takes as input a feature vector of size input_dim and outputs a vector of size ((input_dim + output_dim) * .66).
            nn.Linear(input_dim, round((input_dim + output_dim) * .66)),
            # The activation function for the input layer.
            nn.ReLU(),
            # The first hidden layer. It takes as input the output of the input layer and produces an output of the same size.
            nn.Linear(round((input_dim + output_dim) * .66), round((input_dim + output_dim) * .66)),
            # The activation function for the first hidden layer.
            nn.ReLU(),
            # The output layer. It takes as input the output of the first hidden layer and produces a single output value.
            nn.Linear(round((input_dim + output_dim) * .66), 1)
        )

    def forward(self, X):
        # The forward pass of the network. It takes as input a batch of feature vectors and returns a batch of output values.
        return self.net(X)


# This function calculates the pairwise regret loss given the predicted and true performance differences.
def pairwise_regret_loss(y_pred, y_true):
    # For each predicted performance difference, calculate if it is greater than 0. The result is a tensor of the same size as y_pred, 
    # where each element is 1.0 if the corresponding prediction is greater than 0, and 0.0 otherwise.
    greater = (y_pred > 0).float()
    
    # Calculate the loss. If the predicted performance difference is greater than 0 (i.e., the model predicts that the first algorithm 
    # performs better), then the loss is the positive part of (y_pred - y_true). Otherwise, the loss is the positive part of (y_true - y_pred). 
    # The clamp operation ensures that the loss is non-negative.
    loss = greater * torch.clamp(y_pred - y_true, min=0) + (1 - greater) * torch.clamp(y_true - y_pred, min=0)
    
    # Return the mean loss over all instances in the batch.
    return torch.mean(loss)

def main(data):
    torch.manual_seed(2)
    np.random.seed(2)
    # Load data from files.
    train_features = np.loadtxt(f"{data}/instance-features.txt", delimiter=" ")
    train_performance = np.loadtxt(f'{data}/performance-data.txt', delimiter=" ")
    test_features = np.loadtxt("data/test/instance-features.txt", delimiter=" ")
    test_performance = np.loadtxt('data/test/performance-data.txt', delimiter=" ")

    # Normalize the features using min-max normalization.
    min_features = np.min(train_features, axis=0)
    max_features = np.max(train_features, axis=0)
    eps = 1e-20
    train_features = (train_features - min_features) / (max_features - min_features + eps)
    test_features = (test_features - min_features) / (max_features - min_features + eps)

    # Convert data to the types required by PyTorch.
    train_features = train_features.astype(np.float32)
    train_performance = train_performance.astype(np.float32)
    test_features = test_features.astype(np.float32)
    test_performance = test_performance.astype(np.float32)

    # Compute the list of all possible pairs of algorithms.
    n_algorithms = train_performance.shape[1]
    algo_pairs = [(i, j) for i in range(n_algorithms) for j in range(i + 1, n_algorithms)]

    # Create dataset and dataloader objects for training and testing.
    train_dataset = ASPairwiseDataset(train_features, train_performance, algo_pairs)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = ASPairwiseDataset(test_features, test_performance, algo_pairs)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model and the optimizer.
    model = ASPairwiseRegressor(train_features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    # Training loop with custom loss calculation.
    max_epochs = 100

    for epoch in range(max_epochs):
        total_loss = 0

        for X, y, _ in train_dataloader:
            pred = model(X).squeeze()        
            loss = pairwise_regret_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")

    # Evaluation.
    model.eval()

    # Initialize variables.
    total_loss = 0
    n_correct = 0
    total_cost = 0
    sbs_total_cost = 0
    vbs_total_cost = 0
    votes = np.zeros((test_features.shape[0], n_algorithms))
    
    # With gradients disabled (to save memory and speed up computations), iterate over the test data.
    with torch.no_grad():
        for X, y, pair in test_dataloader:
            # Make a prediction with the model for the current batch of data and compute the pairwise regret loss.
            pred = model(X).squeeze()
            loss = pairwise_regret_loss(pred, y)
            # Add the loss for this batch to the total loss.
            total_loss += loss.item()
    
            # Calculate which algorithm the model thinks is better for each instance in the batch.
            pred_better = (pred > 0).numpy()
            # Count a vote for the predicted better algorithm for each instance.
            votes[:, pair[0]] += pred_better
            votes[:, pair[1]] += 1 - pred_better
    
    # Compute the average loss over all batches.
    avg_loss = total_loss / len(test_dataloader)
    print(votes.shape)
    
    # Determine which algorithm got the most votes for each instance.
    best_predicted_algorithms = np.argmax(votes, axis=1)
    
    # Iterate over the predicted best algorithms and the corresponding true performances.
    for i, predicted_algo in enumerate(best_predicted_algorithms):
        # Find out which algorithm truly performs the best for this instance.
        true_best_algo = np.argmin(test_performance[i])
        # Add the performance of the predicted algorithm to the total cost.
        total_cost += test_performance[i, predicted_algo]
        # Add the performance of the best algorithm on average (SBS) and the best algorithm in hindsight (VBS) to their total costs.
        sbs_total_cost += test_performance[i, np.argmin(train_performance.mean(axis=0))]
        vbs_total_cost += test_performance[i, true_best_algo]
    
        # If the predicted best algorithm is truly the best, count this as a correct prediction.
        if predicted_algo == true_best_algo:
            n_correct += 1
    
    # Compute accuracy and average costs.
    accuracy = n_correct / len(best_predicted_algorithms)
    avg_cost = total_cost / len(best_predicted_algorithms)
    sbs_avg_cost = sbs_total_cost / len(best_predicted_algorithms)
    vbs_avg_cost = vbs_total_cost / len(best_predicted_algorithms)
    # Compute the gap between the SBS and VBS costs.
    sbs_vbs_gap = sbs_avg_cost - vbs_avg_cost


    # Print metrics.
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average cost: {avg_cost:.4f}")
    print(f"SBS average cost: {sbs_avg_cost:.4f}")
    print(f"VBS average cost: {vbs_avg_cost:.4f}")
    print(f"SBS-VBS gap: {sbs_vbs_gap:.4f}")
    return model