import torch
import torch.nn as nn
import numpy as np
from evaluate.evaluate import *
from src.EDSAN import EDSAN, R_3EDSAN, T_3EDSAN
from train import create_kfold_dataloaders, create_custom_dataloaders
from preprocessing.dataset import Dataset
from preprocessing.preprocess import device

def test_model(model, model_path, dataloader_fn, dataset, batch_size, criterion, device, model_name, k=5, mode="train"):
    """Tests the model by loading it from the given path and using a custom or kfold DataLoader."""
    # Load the model from the checkpoint directory
    model.load_state_dict(torch.load(model_path))
    model.to(device)  # Move the model to the appropriate device

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    all_outputs, all_labels = [], []

    # Create the DataLoader using the provided function (KFold or Custom)
    dataloaders = dataloader_fn(dataset, batch_size, k=k, mode=mode)

    with torch.no_grad():  # Ensure no gradients are calculated during testing
        for _, val_dataloader in dataloaders:
            val_loss, fold_outputs, fold_labels = evaluate_model(model, val_dataloader, criterion, device, model_name)

            # Aggregate loss and outputs/labels across folds
            test_loss += val_loss * len(val_dataloader.dataset)
            all_outputs.append(fold_outputs)
            all_labels.append(fold_labels)

    # Compute average loss over all data
    test_loss /= len(dataset)

    # Concatenate all outputs and labels for further metric calculations
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_outputs, all_labels)
    mae, rmse, pcc, snr_pred, tmc, tmc_l, acc = metrics
    plot_hr(all_outputs, all_labels, model_name, sampling_rate=28)
    plot_bvp(all_outputs, all_labels, model_name)
    print({'model_name': model_name}, {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SNR_Pred': snr_pred, 'TMC': tmc, 'TMC_l': tmc_l, 'ACC': acc})




if __name__ == "__main__":

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("test_model")
    with mlflow.start_run():

        videos = torch.rand(56, 4, 192, 64, 64)
        label = torch.rand(56, 192)
        dataset = Dataset(data=videos, labels=label)

        # Assuming the model, dataset, and other parameters are set
        model = EDSAN(n_channels=1, model='thermal').to(device)  # Replace with your model class
        model_path = "model_paths/best_model_Thermal_fold_1.pth"
        batch_size = 8
        k = 2  # Number of folds for KFold or custom splitting
        mode = "eval"  # Mode for testing (usually "eval" for testing)
        criterion = nn.MSELoss()  # Replace with your loss function
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Thermal"  # Name of your model, used for logging/plotting

        # Example: Test using K-Fold DataLoader
        test_model(
            model, model_path, create_kfold_dataloaders, dataset, batch_size, criterion, device, model_name, k=k, mode=mode
        )
