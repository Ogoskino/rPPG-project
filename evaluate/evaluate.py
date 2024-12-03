from evaluate.metrics import *
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from tabulate import tabulate



def evaluate_model(model, dataloader, criterion, device, model_name):
    """Evaluates the model and calculates metrics."""
    model.eval()
    val_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            time_diff = torch.linspace(0, 1, inputs.shape[1]).unsqueeze(0).repeat(inputs.size(0), 1).to(device)
            loss = criterion([outputs, labels, time_diff]) if model_name == 'RTrPPG' else criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)

    val_loss /= len(dataloader.dataset)
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    return val_loss, all_outputs, all_labels

def compute_metrics(outputs, labels, sampling_rate=28):
    """Calculates and returns various metrics."""
    hr_pred, hr_true, snr_pred = get_peak_frequencies_and_snr_batch(outputs, labels, sampling_rate)
    mae = mean_absolute_error(hr_true, hr_pred)
    rmse = root_mean_square_error(hr_true, hr_pred)
    pcc = pearson_correlation(hr_true, hr_pred)
    snr_pred = calculate_mean_snr(snr_pred)
    tmc = calculate_tmc(outputs)
    tmc_l = calculate_tmc(labels)
    tmc_acc = compute_accuracy(tmc_l, tmc)

    return mae, rmse, pcc, snr_pred, tmc, tmc_l, tmc_acc

def save_best_model(model, metrics, best_rmse, file_path):
    """Saves the model if current RMSE is the best."""
    if metrics['RMSE'] < best_rmse:
        torch.save(model.state_dict(), file_path)
        return metrics['RMSE'], metrics
    return best_rmse, None

