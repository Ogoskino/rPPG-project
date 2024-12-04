from evaluate.metrics import *
import torch
import os



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


def save_best_model(model, metrics, best_rmse, filename, folder_path="model_paths"):
    """Saves the model if current RMSE is the best."""

    # Ensure the folder exists, create it if it doesn't
    os.makedirs(folder_path, exist_ok=True)
    
    # Combine folder and filename to get the full path
    file_path = os.path.join(folder_path, filename)

    # Save the model if the current RMSE is the best
    if metrics['RMSE'] < best_rmse:
        torch.save(model.state_dict(), file_path)
        return metrics['RMSE'], metrics
    
    return best_rmse, None



def log_model_summary(model, model_name, input_size, device, folder_path="model_summaries"):
    """
    Logs the summary of the model to MLflow and saves it in a specified folder.
    
    Args:
        model: The PyTorch model instance.
        model_name: The name of the model (e.g., "RGB", "Thermal").
        input_size: The input size tuple to the model.
        device: The device (e.g., "cpu" or "cuda").
        folder_path: The folder path where the summary file will be saved (default: "model_summaries").
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    model = model.to(device)
    
    # Get model summary
    summary_text = str(summary(model, input_size=input_size, device=device.type))

    # Save model summary as a text file in the folder
    summary_file_path = os.path.join(folder_path, f"{model_name}_summary.txt")
    
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    # Log the summary file as an artifact in MLflow
    mlflow.log_artifact(summary_file_path)
    
    print(f"Model summary saved and logged to MLflow: {summary_file_path}")

    
def evaluate_ampnet_model(fusion_model, dataloader, criterion, device, sampling_rate=28):
    """Evaluates the fusion model and computes metrics."""
    fusion_model.eval()
    val_loss = 0.0
    all_outputs, all_labels, all_outputs_rgb = [], [], []

    with torch.no_grad():
        for rgb_inputs, thermal_inputs, labels in dataloader:
            rgb_inputs = rgb_inputs.to(device)
            thermal_inputs = thermal_inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, outputs_rgb, _ = fusion_model(rgb_inputs, thermal_inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item() * rgb_inputs.size(0)

            all_outputs.append(outputs)
            all_outputs_rgb.append(outputs_rgb)
            all_labels.append(labels)

    val_loss /= len(dataloader.dataset)
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_outputs_rgb = torch.cat(all_outputs_rgb).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Compute metrics
    metrics = compute_metrics(all_outputs, all_labels, sampling_rate)
    metrics_rgb = compute_metrics(all_outputs_rgb, all_labels, sampling_rate)

    return val_loss, metrics, metrics_rgb


