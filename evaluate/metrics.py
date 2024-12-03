import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr



def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE).

    Parameters:
    - y_true: numpy array or list, the actual values.
    - y_pred: numpy array or list, the predicted values.

    Returns:
    - mae: float, the Mean Absolute Error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def root_mean_square_error(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE).

    Parameters:
    - y_true: numpy array or list, the actual values.
    - y_pred: numpy array or list, the predicted values.

    Returns:
    - rmse: float, the Root Mean Square Error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def pearson_correlation(y_true, y_pred):
    """
    Calculate the Pearson correlation coefficient.

    Parameters:
    - y_true: numpy array or list, the actual values.
    - y_pred: numpy array or list, the predicted values.

    Returns:
    - pearson_corr: float, the Pearson correlation coefficient.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean of y_true and y_pred
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Covariance between y_true and y_pred
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # Standard deviations
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    
    # Pearson correlation coefficient
    pearson_corr = covariance / (std_true * std_pred)
    
    return pearson_corr




def get_peak_frequencies_and_snr_batch(signals, ground_truths, sampling_rate, lowcut=0.7, highcut=3.5, filter_order=4):
    """
    Calculate the peak frequencies and SNRs of a batch of signals relative to the ground truth.

    Parameters:
    - signals: 2D numpy array of shape (batch_size, num_samples), the predicted signals.
    - ground_truths: 2D numpy array of shape (batch_size, num_samples), the ground truth signals.
    - sampling_rate: float, the sampling rate of the signals (in Hz).
    - lowcut: float, lower bound of the bandpass filter (in Hz).
    - highcut: float, upper bound of the bandpass filter (in Hz).
    - filter_order: int, the order of the Butterworth filter.

    Returns:
    - peak_frequencies_pred: 1D numpy array of shape (batch_size,), the peak frequencies for each predicted signal (in Hz).
    - peak_frequencies_gt: 1D numpy array of shape (batch_size,), the peak frequencies for each ground truth signal (in Hz).
    - snr_values: 1D numpy array of shape (batch_size,), the SNR values for each signal relative to the ground truth (in dB).
    """
    # Ensure signals and ground_truths are 2D arrays
    signals = np.array(signals)
    ground_truths = np.array(ground_truths)

    # Get batch size and number of samples
    batch_size, num_samples = signals.shape

    # Ensure num_samples matches expected number before reshaping
    assert num_samples == 192, "Expected number of samples per signal is 192."

    # Reshape signals and ground_truths to (new_batch, 1792)
    reshaped_signals = signals.reshape((-1, 1792))
    reshaped_ground_truths = ground_truths.reshape((-1, 1792))

    new_batch_size, _ = reshaped_signals.shape

    # Prepare the output arrays
    peak_frequencies_pred = np.zeros(new_batch_size)
    peak_frequencies_gt = np.zeros(new_batch_size)
    snr_values = np.zeros(new_batch_size)

    # Nyquist frequency
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter
    b, a = butter(filter_order, [low, high], btype='band')

    for i in range(new_batch_size):
        # Extract the signal and ground truth
        signal = reshaped_signals[i]
        ground_truth = reshaped_ground_truths[i]

        # Apply the bandpass filter to both signal and ground truth
        filtered_signal = filtfilt(b, a, signal)
        filtered_ground_truth = filtfilt(b, a, ground_truth)

        # Compute the power of the ground truth signal
        signal_power = np.mean(filtered_ground_truth ** 2)

        # Estimate the noise (difference between the ground truth and predicted signal)
        noise_estimate = filtered_ground_truth - filtered_signal
        noise_power = np.mean(noise_estimate ** 2)

        # Compute SNR relative to ground truth power in dB
        snr = signal_power / noise_power
        snr_values[i] = 10 * np.log10(snr)

        # Compute the number of samples and time spacing
        N = len(filtered_signal)
        T = 1.0 / sampling_rate

        # Compute the Fast Fourier Transform (FFT) for both signal and ground truth
        yf_signal = fft(filtered_signal)
        xf_signal = fftfreq(N, T)
        yf_gt = fft(filtered_ground_truth)
        xf_gt = fftfreq(N, T)

        # Compute the power spectrum (magnitude squared of FFT) for both
        power_spectrum_signal = np.abs(yf_signal) ** 2
        power_spectrum_gt = np.abs(yf_gt) ** 2

        # Only take the positive half of the spectrum
        half_N = N // 2
        xf_signal = xf_signal[:half_N]
        power_spectrum_signal = power_spectrum_signal[:half_N]
        xf_gt = xf_gt[:half_N]
        power_spectrum_gt = power_spectrum_gt[:half_N]

        # Identify the peak frequencies for both the predicted signal and the ground truth
        peak_freq_index_signal = np.argmax(power_spectrum_signal)
        peak_frequencies_pred[i] = xf_signal[peak_freq_index_signal]

        peak_freq_index_gt = np.argmax(power_spectrum_gt)
        peak_frequencies_gt[i] = xf_gt[peak_freq_index_gt]

    return peak_frequencies_pred, peak_frequencies_gt, snr_values
    
def calculate_mean_snr(snr_values):
    """
    Calculate the mean SNR from a batch of SNR values.

    Parameters:
    - snr_values: 1D numpy array of SNR values in dB.

    Returns:
    - mean_snr: float, the mean SNR in dB.
    """
    mean_snr = np.mean(snr_values)
    return mean_snr


def evaluate_heart_rate_predictions(hr_truth, hr_pred):
    """
    Evaluate the heart rate predictions against the ground truth values.

    Parameters:
    - hr_truth (np.ndarray): 2D array of ground truth heart rate values (shape: (n_samples, n_time_points)).
    - hr_pred (np.ndarray): 2D array of predicted heart rate values (shape: (n_samples, n_time_points)).

    Returns:
    - metrics (dict): A dictionary containing MAE, RMSE, and PCC.
    """

    # Ensure input arrays are numpy arrays
    hr_truth = np.asarray(hr_truth)
    hr_pred = np.asarray(hr_pred)

    # Validate that shapes match
    if hr_truth.shape != hr_pred.shape:
        raise ValueError("Ground truth and predicted values must have the same shape.")


    # Calculate per-person metrics
    mae_per_person = np.array([mean_absolute_error(hr_truth[i], hr_pred[i]) for i in range(hr_truth.shape[0])])
    rmse_per_person = np.array([np.sqrt(mean_squared_error(hr_truth[i], hr_pred[i])) for i in range(hr_truth.shape[0])])
    pcc_per_person = np.array([pearsonr(hr_truth[i], hr_pred[i])[0] for i in range(hr_truth.shape[0])])

    # Calculate average metrics across all individuals
    avg_mae = np.mean(mae_per_person)
    avg_rmse = np.mean(rmse_per_person)
    avg_pcc = np.mean(pcc_per_person)

    # Return results in a dictionary
    metrics = {
        'Avg MAE': avg_mae,
        'Avg RMSE': avg_rmse,
        'Avg PCC': avg_pcc,
    }

    return metrics


def calculate_tmc(signals, reshape_size=28):
    batch_size, signal_length = signals.shape
    
    # Step 1: Reshape the signals to [28, -1, 192]
    reshaped_signals = signals.reshape(reshape_size, -1, signal_length)  # [28, batch_segments, 192]

    # Initialize an array to hold the TMC for each batch in the reshaped signal
    tmc_values = []
    
    # Loop through each of the 28 batches
    for batch in reshaped_signals:
        windowed_pulses = []
        template_pulses = []
        correlations = []
        
        # Loop through each signal within the current batch
        for signal in batch:
            # Step 2: Detect Peaks in each signal
            peaks, _ = find_peaks(signal)

            # Step 3: Calculate Median Beat-to-Beat Interval
            beat_intervals = np.diff(peaks)
            if len(beat_intervals) == 0:
                continue  # If no peaks, skip this signal
            median_interval = np.median(beat_intervals)

            # Step 4: Extract Individual Pulses
            pulses = []
            for peak in peaks:
                start = max(0, int(peak - median_interval // 2))
                end = min(len(signal), int(peak + median_interval // 2))
                pulses.append(signal[start:end])

            # Ensure all pulses have the same length by truncating or padding
            pulse_length = int(median_interval)
            pulses = [np.pad(p, (0, max(0, pulse_length - len(p))), mode='constant')[:pulse_length] for p in pulses]

            windowed_pulses.append(pulses)

            # Step 5: Create Template Pulse
            template_pulse = np.mean(pulses, axis=0)
            template_pulses.append(template_pulse)

            # Step 6: Calculate Correlations
            pulse_correlations = [np.corrcoef(p, template_pulse)[0, 1] for p in pulses]
            correlations.append(np.mean(pulse_correlations))

        # Calculate the average correlation for this batch
        if len(correlations) > 0:
            tmc_batch_average = np.mean(correlations)
            tmc_values.append(tmc_batch_average)

    # Step 7: Calculate the average TMC across all 28 batches
    if len(tmc_values) > 0:
        tmc_average = np.mean(tmc_values)
    else:
        tmc_average = 0  # Default if no valid correlations are found

    # Return the average TMC
    return tmc_average