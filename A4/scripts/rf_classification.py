from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main(data):
    # Set a fixed seed for numpy's random number generator for reproducibility
    np.random.seed(2)

    # Load the features and performance data from text files
    train_features = np.loadtxt(f"{data}instance-features.txt", delimiter=" ")
    train_performance = np.loadtxt(f'{data}performance-data.txt', delimiter=" ")

    # Split the loaded data into training and validation sets
    # 80% of the data is used for training and 20% for validation
    train_features, valid_features, train_performance, valid_performance = train_test_split(train_features, train_performance, test_size=0.2)

    # Initialize a random forest classifier
    rf_classifier = RandomForestClassifier()

    # Initialize a MinMaxScaler to scale the feature data to the range [0, 1]
    scaler = MinMaxScaler()

    # Fit the scaler to the training feature data and transform the training and validation feature data
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)

    # Identify the best algorithm for each instance in the training and validation data
    # This is done by finding the algorithm with the minimum performance (argmin) for each instance
    train_best_algo = np.argmin(train_performance, axis=1)
    valid_best_algo = np.argmin(valid_performance, axis=1)

    # Train the random forest classifier using the scaled training feature data and the identified best algorithms
    rf_classifier.fit(train_features, train_best_algo)

    # Evaluate the trained classifier on the validation data and print its accuracy
    clf_accuracy = rf_classifier.score(valid_features, valid_best_algo)
    print(f"Classification approach accuracy: {clf_accuracy:.4f}")

    # Return the trained classifier and the fitted scaler
    return rf_classifier, scaler
