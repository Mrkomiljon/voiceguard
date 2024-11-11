import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from model import RawNet
import yaml
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model configuration
dir_yaml = 'model_config_RawNet.yaml'
with open(dir_yaml, 'r') as f_yaml:
    parser1 = yaml.safe_load(f_yaml)
# Load your model (assuming parser1 is already defined and configured)
model = RawNet(parser1['model'], device)
model.load_state_dict(torch.load("path_to_best_model.pth", map_location=device))

# Apply pruning to convolutional and linear layers
def apply_pruning(module, pruning_amount=0.5):
    for name, layer in module.named_modules():
        if isinstance(layer, torch.nn.Conv1d) or isinstance(layer, torch.nn.Linear):
            prune.l1_unstructured(layer, name="weight", amount=pruning_amount)

# Apply pruning to the model
apply_pruning(model)

# Optionally, remove pruning remainders to keep the model clean
def remove_pruning(module):
    for name, layer in module.named_modules():
        if isinstance(layer, torch.nn.Conv1d) or isinstance(layer, torch.nn.Linear):
            prune.remove(layer, 'weight')

# Remove the pruning remainders
remove_pruning(model)

# Save the pruned model
torch.save(model.state_dict(), "pruned_model.pth")
print("Pruned model saved.")

