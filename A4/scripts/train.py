import argparse
import regression
import classification
import pairwise_classification
import rf_classification
import classification_adv
import torch
import pickle

def main():
    # Initialize an argument parser
    parser = argparse.ArgumentParser(description="Train an AS model and save it to file")

    # Add expected arguments
    parser.add_argument("--model-type", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    parser.add_argument("--save", type=str, required=True, help="Save the trained model (and any related info) to a .pt file")
    
    # Parse the arguments
    args = parser.parse_args()

    # Print out the operation details
    print(f"\nTraining a {args.model_type} model on {args.data}, and save it to {args.save}")

    # Depending on the model type, call the appropriate main() function to train the model
    if(args.model_type == 'regresion_nn'):
        model = regression.main(args.data)
    elif(args.model_type == 'classification_nn'):
        model = classification.main(args.data)
    elif(args.model_type == 'classification_nn_cost'):
        model = classification_adv.main(args.data)
    elif(args.model_type == 'pairwise_classification_nn'):
        model = pairwise_classification.main(args.data)
    elif(args.model_type == 'rf_classification_nn'):
        # For RandomForestClassifier, not only the model but also the scaler is returned
        model, scalar = rf_classification.main(args.data)
        # Save the trained model and scaler to pickle files
        with open(args.save, "wb") as output_file:
            pickle.dump(model, output_file)
        with open(r"models/scalar.pickle", "wb") as output_scalar:
            pickle.dump(scalar, output_scalar)
        print(f"\nTraining finished")
        return model

    # For other models, save the model's state dict to a .pt file
    torch.save(model.state_dict(), args.save)
    # print results
    print(f"\nTraining finished")


if __name__ == "__main__":
    main()
