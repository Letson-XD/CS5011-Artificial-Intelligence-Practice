import argparse
import torch
import numpy as np
import regression
import classification
import classification_adv
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AS model on a test set")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    
    args = parser.parse_args()

    print(f"\nLoading trained model {args.model} and evaluating it on {args.data}")
    
    # load the given model, make predictions on the given dataset and evaluate the model's performance. Your evaluation should report four evaluation metrics: avg_loss, accuracy, avg_cost, sbs_vbs_gap (as listed below)
    # you should also calculate the average cost of the SBS and the VBS
    avg_loss = np.inf # the average loss value across the given dataset
    accuracy = 0 # classification accuracy 
    avg_cost = np.inf # the average cost of the predicted algorithms on the given dataset
    sbs_vbs_gap = np.inf # the SBS-VBS gap of your model on the given dataset
    sbs_avg_cost = np.inf # the average cost of the SBS on the given dataset 
    vbs_avg_cost = np.inf # the average cost of the VBS on the given dataset
    test_features = np.loadtxt(f"{args.data}/instance-features.txt")
    test_performance = np.loadtxt(f'{args.data}/performance-data.txt')
    # Load the training performance data.
    train_features = np.loadtxt('data/train/instance-features.txt')
    train_performance = np.loadtxt('data/train/performance-data.txt')
    # Convert data to the types required by PyTorch
    test_features = test_features.astype(np.float32)
    test_features = test_features.astype(np.float32)
    train_features = train_features.astype(np.float32)
    train_performance = train_performance.astype(np.float32)

    # If the specified model is a pre-trained random forest model.
    if(args.model == 'models/part3_rf.pickle'):
        # Load the model and the scaler used to scale the training data from a file.
        with open(args.model, "rb") as input_model:
            model = pickle.load(input_model)
        with open(r"models/scalar.pickle", "rb") as input_scalar:
            scalar = pickle.load(input_scalar)

        # Scale the test features using the same scaler used on the training data.
        test_features = scalar.transform(test_features)

        # Use the model to make predictions on the test data.
        pred_class = model.predict(test_features)

        # The true class is the algorithm that performs the best on each test instance.
        true_class = test_performance.argmin(axis=1)

        # Compute the accuracy as the fraction of correct predictions.
        n = test_features.shape[0]
        accuracy = (pred_class == true_class).sum() / n

        # Compute the average cost as the mean performance of the predicted best algorithm for each instance.
        test_performance = torch.tensor(test_performance)
        pred_class = torch.tensor(pred_class)
        true_class = torch.tensor(true_class)
        f = test_performance.gather(1, pred_class.long().unsqueeze(1))
        avg_cost = torch.mean(f)

        # Compute the VBS cost as the mean performance of the best algorithm for each instance.
        vbs = test_performance.gather(1, true_class.long().unsqueeze(1))
        vbs_avg_cost = torch.mean(vbs)

        # Compute the SBS cost as the mean performance of the best algorithm on average.
        test_avgs = torch.mean(test_performance, dim=0)
        sbs = test_avgs.argmin()
        sbs_avg_cost = test_avgs[sbs.item()]

        # Compute the SBS-VBS gap.
        sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)

        # Loss is not meaningful for the random forest model.
        avg_loss = -1

        # Print the final results.
        print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")
        return

    # Scale the test features to the range of the training features.
    eps = 1e-20
    test_features = (test_features - np.min(train_features, axis=0)) / (np.max(train_features, axis=0) - np.min(train_features, axis=0) + eps)

    # If the model is one of the neural network models, create an appropriate test dataset and model and load the model parameters from a file.
    # Then create a dataloader for the test data and set the model to evaluation mode.
    if(args.model == 'models/part1.pt'):
        test_dataset = regression.ASRegressionDataset(test_features, test_performance)
        model = regression.ASRegressor(test_features.shape[1], test_performance.shape[1])
        loss_function = nn.MSELoss(reduction="mean")
    elif(args.model == 'models/part2_basic.pt'):
        test_labels = np.argmin(test_performance, axis=1)
        test_dataset = classification.ASClassificationDataset(test_features, test_labels)
        model = classification.ASClassifier(test_features.shape[1], len(np.unique(test_labels)) + 1)
        loss_function = nn.CrossEntropyLoss(reduction="mean")
    elif(args.model == 'models/part2_advanced.pt'):
        test_labels = np.argmin(test_performance, axis=1)
        test_dataset = classification_adv.ASRegretDataset(test_features, test_labels, test_performance)
        model = classification_adv.ASClassifier(test_features.shape[1], len(np.unique(test_labels)) + 1)
        loss_function = classification_adv.loss_function
    model.load_state_dict(torch.load(args.model))
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    total_loss = 0
    total_cost = 0
    total_correct = 0
    n_samples = 0

    with torch.no_grad():
        if (args.model != 'models/part2_advanced.pt'):
            for X, y in dataloader:
                # Forward pass: get the predictions for the current batch
                pred = model(X)

                # Calculate the loss on the current batch
                loss = loss_function(pred, y)

                # Get the index of the minimum predicted cost for each instance
                if(args.model == 'models/part1.pt'):
                    pred_best_algo = torch.argmin(pred, dim=1)
                    true_best_algo = torch.argmin(y, dim=1)
                    pred_cost = torch.gather(y, 1, pred_best_algo.view(-1, 1)).squeeze()
                else:
                    # Get the index of the minimum predicted cost for each instance
                    pred_best_algo = torch.argmax(pred, dim=1)
                    true_best_algo = y
                    pred_cost = test_performance[np.arange(len(pred_best_algo))+n_samples, pred_best_algo.numpy()]

                # Update statistics
                total_loss += loss.item()
                total_cost += pred_cost.sum().item()
                total_correct += torch.sum(pred_best_algo == true_best_algo).item()
                n_samples += X.size(0)
        else:
            for X, y, perf in dataloader:
                # Forward pass: get the predictions for the current batch
                pred = model(X)

                # Calculate the loss on the current batch
                loss = loss_function(pred, y, perf)

                # Get the index of the minimum predicted cost for each instance
                pred_best_algo = torch.argmax(pred, dim=1)

                true_best_algo = y

                pred_cost = test_performance[np.arange(len(pred_best_algo))+n_samples, pred_best_algo.numpy()]

                # Update statistics
                total_loss += loss.item()
                total_cost += pred_cost.sum().item()
                total_correct += torch.sum(pred_best_algo == true_best_algo).item()
                n_samples += X.size(0)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / n_samples
    avg_cost = total_cost / n_samples
    vbs_avg_cost = np.mean(np.min(test_performance, axis=1))
    sbs_avg_cost = np.min(np.mean(test_performance, axis=0))
    sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)
    # print results
    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")




if __name__ == "__main__":
    main()
