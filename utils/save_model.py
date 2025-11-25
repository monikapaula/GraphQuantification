import os
import torch

def save_model(model, config, dataset_name, save_dir="saved_models"):
    model_name = config.get('name', 'model')

    filename = f"{model_name}_{dataset_name}.pth"
    save_dir = "saved_models"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), save_path)