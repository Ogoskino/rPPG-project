import torch
import numpy as np
from dataloader import load_iBVP_dataset

def normalize_array(arr):
    """
    Normalize a NumPy array to the range [0, 1].

    Parameters:
    arr (np.ndarray): The input array to normalize.

    Returns:
    np.ndarray: The normalized array with values between 0 and 1.
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr


def preprocess_data(rgb_face, thermal_face, label):
    thermal_array = np.expand_dims(thermal_face, axis=-1)
    rgb_faces = np.array(rgb_face)/rgb_face.max()
    thermal_array = np.array(thermal_array)/thermal_array.max()
    videos = np.concatenate((rgb_faces, thermal_array), axis=-1)
    bvps = label
    a = np.array(videos)
    b = normalize_array(np.array(bvps))
    return a, b

def extract_segments(data, ppg_signals, segment_length=192):
    segments = []
    ppg_segments = []
    for i in range(data.shape[0]):  # Loop over subjects
        for start_frame in range(0, data.shape[1] - segment_length + 1, segment_length):
            end_frame = start_frame + segment_length
            segments.append(data[i, start_frame:end_frame])
            ppg_segments.append(ppg_signals[i, start_frame:end_frame])
    return torch.stack(segments), torch.stack(ppg_segments)


def to_tensor(a, b):
    videos = torch.tensor(a, dtype=torch.float32)
    labels = torch.tensor(b, dtype=torch.float32)
    return videos, labels




if __name__ == "__main__":
    dataset_path = r'C:\Users\jkogo\OneDrive\Desktop\PHD resources\datasets\iBVP_Dataset'
    rgb_face, thermal_face, label = load_iBVP_dataset(dataset_path)
    a, b = preprocess_data(rgb_face, thermal_face, label)
    videos, labels = to_tensor(a,b)
    video_chunks, label_chunks = extract_segments(videos, labels)
    print(video_chunks.shape)  # Should be [num_chunks, chunk_size, height, width, channels]
    print(label_chunks.shape)  # Should be [num_chunks, chunk_size]