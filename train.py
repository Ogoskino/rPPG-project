from src.iBVPNet import iBVPNet
from src.EDSAN import EDSAN
from src.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from src.RTrPPG import N3DED64
from evaluate.loss import *
import mlflow
from mlflow.models import set_signature
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from tabulate import tabulate
import torch.optim as optim
from preprocessing.dataset import Dataset
from evaluate.evaluate import *
from preprocessing.preprocess import device
from preprocessing.dataloader import load_iBVP_dataset
from preprocessing.preprocess import *

os.environ["MLFOW_TRACKIN_URI"] = "https://dagshub.com/jkogoskino/ml_project_repo.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "jkogoskino"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9fd2591fb0fdf03c2e97f8ee1775575b7bb6323c"



def initialize_model(model_class, model_name, criterion_classes, optimizer_classes, scheduler_classes, model_idx, device):
    """Initializes model, criterion, optimizer, and scheduler."""
    if model_name == 'iBVPNet':
        model = model_class(in_channels=3, frames=192, debug=True).to(device)
    if model_name == 'Thermal':
        model = model_class(n_channels=1, model='thermal').to(device)
    else:
        model = model_class().to(device)

    if model_name == 'RTrPPG':
        criterion = criterion_classes[model_idx](Lambda=1.32)
        optimizer = optimizer_classes[model_idx](model.parameters(), lr=0.00044)
    else:
        criterion = criterion_classes[model_idx]()
        optimizer = optimizer_classes[model_idx](model.parameters(), lr=0.0001)

    scheduler = scheduler_classes[model_idx]
    if scheduler:
        scheduler = scheduler(optimizer, step_size=2, gamma=0.95)
    
    return model, criterion, optimizer, scheduler



def train_one_epoch(model, dataloader, criterion, optimizer, device, model_name, batch_size):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        time_diff = torch.linspace(0, 1, inputs.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(device)
        loss = criterion([outputs, labels, time_diff]) if model_name == 'RTrPPG' else criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)

def train_and_evaluate(models, n_splits, model_names, device, data, label):
    """Performs k-fold cross-validation for multiple models."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results_dict = {name: [] for name in model_names}
    average_results_dict = {name: {} for name in model_names}

    for model_idx, model_class in enumerate(models):
        model_name = model_names[model_idx]
        dataset = Dataset(data=data, labels=label)

        print(f"\nTraining Model {model_name} with {n_splits}-fold Cross-Validation")
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=8, shuffle=False)

            model, criterion, optimizer, scheduler = initialize_model(
                model_class, model_name, criterion_classes, optimizer_classes, scheduler_classes, model_idx, device
            )

            best_val_rmse = float('inf')
            best_metrics = {}

            for epoch in range(1):
                train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, model_name, batch_size=8)
                val_loss, all_outputs, all_labels = evaluate_model(model, val_dataloader, criterion, device, model_name)

                metrics = compute_metrics(all_outputs, all_labels)
                mae, rmse, pcc, snr_pred, tmc, tmc_l, acc = metrics

                if scheduler:
                    scheduler.step()

                print(f"Model {model_name}, Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                if rmse < best_val_rmse:
                    best_val_rmse, best_metrics = save_best_model(
                        model, {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SNR_Pred': snr_pred, 'TMC': tmc, 'TMC_l': tmc_l, 'ACC': acc},
                        best_val_rmse, f'best_model_{model_name}_fold_{fold + 1}.pth'
                    )

            fold_results.append(best_metrics)

        # Aggregate results
        average_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
        fold_results_dict[model_name] = fold_results
        average_results_dict[model_name] = average_results

    return fold_results_dict, average_results_dict





if __name__ == "__main__":
    model_classes = [EDSAN, EDSAN, iBVPNet, PhysNet_padding_Encoder_Decoder_MAX, N3DED64]  # Model classes instead of instances
    criterion_classes = [nn.MSELoss, nn.MSELoss, CosineSimilarityLoss, nn.MSELoss, NPSNR]
    optimizer_classes = [optim.Adam, optim.Adam, optim.Adam, optim.Adam, optim.Adam]
    scheduler_classes = [None, None,optim.lr_scheduler.StepLR, None, None]  # Optional
    model_names = ['RGB', 'Thermal', 'iBVPNet', 'PhysNet','RTrPPG']


    #dataset_path = r'C:\Users\jkogo\OneDrive\Desktop\PHD resources\datasets\iBVP_Dataset'
    #rgb_face, thermal_face, label = load_iBVP_dataset(dataset_path)
    #a, b = preprocess_data(rgb_face, thermal_face, label)
    #videos, labels = to_tensor(a,b)



    videos = torch.rand(56, 4, 192, 64, 64)
    label = torch.rand(56, 192)
    #video_chunks, label_chunks = extract_segments(videos, label)
    print(videos.shape)  # Should be [num_chunks, chunk_size, height, width, channels]
    print(label.shape)  # Should be [num_chunks, chunk_size]


    fold_results, avg_results = train_and_evaluate(
        model_classes, n_splits=2, model_names=model_names, device=device,data=videos, label=label
    )

    # Print Results
    print("\nSummary of Results Across All Models and Folds:")
    for model_name in model_names:
        print(f"\nResults for {model_name}:")
        print(tabulate(fold_results[model_name], headers="keys"))

    print("\nAverage Results for Each Model Across All Folds:")
    print(tabulate(avg_results.items(), headers=["Model Name", "Metrics"]))