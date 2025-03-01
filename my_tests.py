from evaluation.post_process import *
from evaluation.metrics import *
import torch


predictions = torch.randn(200, 192)
labels = torch.randn(200, 192)
print(predictions.shape)
print(predictions.shape[0])
hr_labels, hr_preds, SNRs, maccs = [], [], [], []
for i in range(predictions.shape[0]):
    hr_label, hr_pred, SNR, macc = calculate_metric_per_video(predictions[i], labels[i], fs=28, diff_flag=True, use_bandpass=True, hr_method='FFT')
    hr_labels.append(hr_label)
    hr_preds.append(hr_pred)
    SNRs.append(SNR)
    maccs.append(macc)



print(hr_labels)