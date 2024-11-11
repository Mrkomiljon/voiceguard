import torch
from model import RawNet  # Assuming RawNet is defined in model.py
import yaml
from torch import Tensor
import librosa
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model configuration (same as used during training)
dir_yaml = 'model_config_RawNet.yaml'
with open(dir_yaml, 'r') as f_yaml:
    parser1 = yaml.safe_load(f_yaml)

# Initialize the model structure
model = RawNet(parser1['model'], device)

# Load the quantized model weights
quantized_model_path = "quantized_model.pth"
model.load_state_dict(torch.load(quantized_model_path, map_location=device))
model.eval()  # Set to evaluation mode
print("Quantized model loaded for inference.")

# Example inference function
def infer(model, audio_path):
    # Load and preprocess the audio file
    audio, sr = librosa.load(audio_path, sr=24000)
    audio_tensor = Tensor(audio).to(device)
    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output_binary, output_multi = model(audio_tensor)
        _, pred_binary = torch.max(output_binary, dim=1)
        _, pred_multi = torch.max(output_multi, dim=1)

    # Output results
    print(f"Binary Prediction (Real vs. Fake): {pred_binary.item()}")
    print(f"Multi-Class Prediction (Class): {pred_multi.item()}")

# Run inference on a sample audio file
audio_path = "path_to_audio_file.wav"
infer(model, audio_path)
