import torch
import torch.nn as nn
import numpy as np
from evaluate.evaluate import *
from src.EDSAN import EDSAN, R_3EDSAN, T_3EDSAN
from src.iBVPNet import iBVPNet
from src.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from src.RTrPPG import N3DED64
from src.AMPNET import *
from train import create_kfold_dataloaders, create_custom_dataloaders
from preprocessing.dataset import *
from preprocessing.preprocess import device


def log_and_visualize_results(all_outputs, metrics, all_labels, model_name, sampling_rate=28):
    """
    Processes and logs the results, generates metrics, and creates visualizations for the model's outputs.

    Args:
        all_outputs (list or np.ndarray): Model predictions across all folds or splits.
        all_labels (list or np.ndarray): Ground truth labels across all folds or splits.
        model_name (str): The name of the model (used in visualizations and logs).
        sampling_rate (int): The sampling rate for heart rate calculation (default is 28).

    Returns:
        dict: A dictionary containing the computed metrics (MAE, RMSE, PCC, SNR, TMC, TMC_l, ACC).
    """
    # Concatenate outputs and labels if they are in list form
    if isinstance(all_outputs, list):
        all_outputs = np.concatenate(all_outputs, axis=0)
    if isinstance(all_labels, list):
        all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    if model_name == "AMPNet":
        mae, rmse, pcc, snr_pred, tmc, tmc_l, acc = metrics
    else:
        metrics = compute_metrics(all_outputs, all_labels)
        mae, rmse, pcc, snr_pred, tmc, tmc_l, acc = metrics

    # Generate plots
    plot_hr(all_outputs, all_labels, model_name, sampling_rate=sampling_rate)
    plot_bvp(all_outputs, all_labels, model_name)

    # Prepare metrics dictionary
    metrics_dict = {
        'MAE': mae,
        'RMSE': rmse,
        'PCC': pcc,
        'SNR_Pred': snr_pred,
        'TMC': tmc,
        'TMC_l': tmc_l,
        'ACC': acc
    }

    # Print and log metrics
    print(model_name, metrics_dict)
    mlflow.log_metrics(metrics_dict)

    return metrics_dict



def test_model(model, model_path, dataloader_fn, dataset, batch_size, criterion, device, model_name, k=5, mode="train"):
    """Tests the model by loading it from the given path and using a custom or kfold DataLoader."""
    # Load the model from the checkpoint directory

    if model_name == "AMPNet":
        # Initialize models and fusion model
        rgb_models, thermal_models = load_models(R_3EDSAN, T_3EDSAN, device)
        model = AMPNet(rgb_models, thermal_models, normalization=True).to(device)
    else:
        model = model


    model.load_state_dict(torch.load(model_path))
    model.to(device)  # Move the model to the appropriate device

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    all_outputs, all_labels = [], []

    # Create the DataLoader using the provided function (KFold or Custom)
    dataloaders = dataloader_fn(dataset, batch_size, k=k, mode=mode)

    with torch.no_grad():  # Ensure no gradients are calculated during testing
        for _, val_dataloader in dataloaders:
            if model_name == "AMPNet":
                val_loss, metrics, metrics_rgb, all_outputs, all_outputs_rgb, all_labels = evaluate_ampnet_model(model, val_dataloader, criterion, device)
            else:
                val_loss, fold_outputs, fold_labels = evaluate_model(model, val_dataloader, criterion, device, model_name)

                # Aggregate loss and outputs/labels across folds
                test_loss += val_loss * len(val_dataloader.dataset)
                all_outputs.append(fold_outputs)
                all_labels.append(fold_labels)

    # Compute average loss over all data
    test_loss /= len(dataset)
    log_and_visualize_results(all_outputs, metrics, all_labels, model_name, sampling_rate=28)





if __name__ == "__main__":
    dataset_division = "random"
    model_name = "AMPNet"  # Name of your model, used for logging/plotting
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("test_model")
    with mlflow.start_run():
        videos = torch.rand(56, 4, 192, 64, 64)
        label = torch.rand(56, 192)
        
        if model_name == "AMPNet":
            fusion_dataloader = create_ampnet_dataloader(videos, label, batch_size=8, shuffle=True)
            dataset = fusion_dataloader.dataset
            model_path = "model_paths/best_model_AMPNet_fold_1.pth"
        else:
            dataset = Dataset(data=videos, labels=label)
            model_path = "model_paths/best_model_{model_name}_fold_1.pth"
        if model_name == "Thermal":
            model = EDSAN(n_channels=1, model='thermal')
        if model_name == "RGB":
            model = EDSAN()
        if model_name == "iBVPNet":
             model = iBVPNet(in_channels=3, frames=192, debug=True)
        if model_name == "PhysNet":
             model = PhysNet_padding_Encoder_Decoder_MAX()
        if model_name == "PhysNet":  
            model = N3DED64().to(device)
        model.to(device)
        
        batch_size = 8
        k = 2  # Number of folds for KFold or custom splitting
        mode = "eval"  # Mode for testing (usually "eval" for testing)
        criterion = nn.MSELoss()  # Replace with your loss function
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        if dataset_division == "random":
            test_model(
                model, model_path, create_kfold_dataloaders, dataset, batch_size, criterion, device, model_name, k=k, mode=mode
            )
        else:
                        test_model(
                model, model_path, create_custom_dataloaders, dataset, batch_size, criterion, device, model_name, k=k, mode=mode
            )
