import torch
from evaluate.evaluate import *
from src.EDSAN import EDSAN, R_3EDSAN, T_3EDSAN
from src.AMPNET import *
from preprocessing.dataset import *
from preprocessing.preprocess import device

def infer_model(model_path, data, batch_size=32):
    """
    Perform inference using a pre-trained AMPNet model with RGB and thermal data.
    
    Args:
        model_path (str): Path to the saved model checkpoint.
        device (torch.device): Device to run the inference on (e.g., 'cpu' or 'cuda').
        data (numpy.ndarray or torch.Tensor): Input data of shape (num_samples, 6, height, width, depth).
        batch_size (int): Batch size for the DataLoader.
    
    Returns:
        numpy.ndarray: Model outputs for all the input data.
    """
    # Ensure the input data is a PyTorch tensor
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # Define the DataLoader
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize models and fusion model
    rgb_models, thermal_models = load_models(R_3EDSAN, T_3EDSAN, device)
    model = AMPNet(rgb_models, thermal_models, normalization=True).to(device)

    # Load the model weights from the checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)  # Move the model to the appropriate device

    model.eval()  # Set the model to evaluation mode
    all_outputs = []

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            # Unpack the batch and split into RGB and thermal channels
            batch = batch[0]  # TensorDataset returns data as tuples
            rgb_data = batch[:, :3, :, :, :].to(device)  # Extract RGB channels
            thermal_data = batch[:, 3:, :, :, :].to(device)  # Extract thermal channels

            # Forward pass
            outputs, _, _ = model(rgb_data, thermal_data)
            all_outputs.append(outputs.cpu().numpy())  # Collect outputs as NumPy arrays

    # Concatenate all batch outputs into a single array
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs


if __name__ == "__main__":
    model_path = "model_paths/AMPNet.pth"
    data =  np.random.rand(56, 4, 192, 64, 64)
    outputs = infer_model(model_path, data)
    print(outputs)