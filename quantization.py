import torch
from model import RawNet  # Assuming RawNet is defined in model.py
import yaml
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model configuration
dir_yaml = 'model_config_RawNet.yaml'
with open(dir_yaml, 'r') as f_yaml:
    parser1 = yaml.safe_load(f_yaml)
# Load the pruned model
model = RawNet(parser1['model'], device)
model.load_state_dict(torch.load("pruned_model.pth", map_location=device))

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_pruned_model.pth")
print("Quantized and pruned model saved, size reduced.")