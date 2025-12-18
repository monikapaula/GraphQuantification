import os
import torch
from pathlib import Path

def save_model(model, config, dataset_name, split_name= None):
    BASE_DIR = Path(__file__).resolve().parent.parent
    SAVE_DIR = BASE_DIR / "saved_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_name = config.get('name', 'model')

    filename = f"{model_name}_{dataset_name}.pth"

    save_path = SAVE_DIR / filename

    config = {
        'model_state_dict': model.state_dict(),
        'model_config': config,
        'dataset_name': dataset_name,
        'split_name': split_name,
    }

    torch.save(config, save_path)