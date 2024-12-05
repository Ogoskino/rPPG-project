import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import os
import mlflow




# Bland-Altman plot function
def bland_altman_plot(hr_true, hr_pred, model_name):
    mean_hr = np.mean([hr_true, hr_pred], axis=0)
    diff_hr = hr_true - hr_pred
    mean_diff = np.mean(diff_hr)
    std_diff = np.std(diff_hr)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_hr, diff_hr, color='red', alpha=0.5)
    plt.axhline(mean_diff, color='teal', linestyle='--', label=f'mean diff: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96*std_diff, color='black', linestyle='--', label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    plt.axhline(mean_diff - 1.96*std_diff, color='black', linestyle='--', label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    plt.xlabel(f'Mean of HR(True) and HR({model_name})')
    plt.ylabel(f'(HR True - HR {model_name})')
    plt.title(f'{model_name}')
    plt.legend()
    #plt.grid(True)
    plt.savefig(f'bland_altman_{model_name}.png')
    plt.show()

# Scatter plot function for HR(true) vs HR(pred)
def scatter_plot(hr_true, hr_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(hr_true, hr_pred, color='red', alpha=0.5)
    pcc_value = pearsonr(hr_true, hr_pred)[0]
    plt.plot([min(hr_true), max(hr_true)], [min(hr_true), max(hr_true)], color='green', linestyle='--', label=f'Identity line\nr: {pcc_value:.2f}')
    plt.xlabel('HR(True)')
    plt.ylabel(f'HR({model_name})')
    plt.title(f'{model_name}')
    plt.legend()
    #plt.grid(True)
    plt.savefig(f'scatter_plot_{model_name}.png')
    plt.show()


def plot_heart_rate(hr_true, hr_pred, model_name, save_dir='./plots/HRs'):
    """
    This function plots the ground truth vs predicted heart rate and logs it as an artifact.

    Args:
    - hr_true (numpy.ndarray): Ground truth heart rate values.
    - hr_pred (numpy.ndarray): Predicted heart rate values.
    - model_name (str): The name of the model being used.
    - save_dir (str): Directory to save the plots. Default is current directory.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plotting the heart rate
    plt.figure(figsize=(8, 6))
    plt.plot(hr_true * 60, label='Ground-Truth', color='black', marker='o', markersize=3)
    plt.plot(hr_pred * 60, label=model_name, color='teal', linestyle='--', marker='o', markersize=3)
    plt.xlabel('Time')
    plt.ylabel('Heart Rate in BPM')
    plt.legend()

    # Save the plot in the specified folder
    plot_filename = os.path.join(save_dir, f'hr_true_vs_pred_{model_name}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    # Log the plot as an artifact using MLflow
    mlflow.log_artifact(plot_filename)

def plot_bvp_signals(fold_labels, fold_outputs, model_name, save_dir='./plots/bvps'):
    """
    This function plots the BVP signal amplitude for the first two entries and logs them as artifacts.

    Args:
    - fold_labels (list): List containing true BVP signals for each fold.
    - fold_outputs (list): List containing predicted BVP signals for each fold.
    - model_name (str): The name of the model being used.
    - save_dir (str): Directory to save the plots. Default is current directory.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plotting the BVP signals for the first two entries
    for i in range(2):
        plt.figure(figsize=(12, 3))
        plt.plot(fold_labels[i], label='Ground-Truth', color='black')
        plt.plot(fold_outputs[i], label=model_name, color='teal', linestyle='--')
        plt.xlabel('Frames')
        plt.ylabel('BVP Signal Amplitude')
        plt.legend()

        # Save the plot in the specified folder
        plot_filename = os.path.join(save_dir, f'pred_true_bvp_{i+1}_{model_name}.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free up memory

        # Log the plot as an artifact using MLflow
        mlflow.log_artifact(plot_filename)

# Example usage:
# Assuming you are in an active MLflow run
# plot_heart_rate(hr_true, hr_pred, 'R-3EDSAN', './plots')
# plot_bvp_signals(fold_labels, fold_outputs, 'AMPNet', './plots')
