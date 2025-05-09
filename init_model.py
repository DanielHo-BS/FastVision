import torch
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, 
    ResNet50_Weights, 
    VGG16_Weights, 
    ViT_B_16_Weights
)

def setup_env(device_name):
    """
    Set up the computing environment based on the device name.
    
    Args:
        device_name (str): Name of the device ('cpu' or 'gpu')
        
    Returns:
        torch.device: The device to use for computations
    """
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Invalid device: {device_name}")

def init_model(model_name, device):
    """
    Initialize a model with appropriate weights based on model name.
    
    Args:
        model_name (str): Name of the model to initialize
        device (torch.device): Device to move the model to
        
    Returns:
        torch.nn.Module: Initialized model on the specified device
    """
    if model_name == "resnet18":
        return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    elif model_name == "resnet50":
        return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    elif model_name == "vgg16":
        return models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    elif model_name == "vision_transformer":
        return models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    else:
        raise ValueError(f"Invalid model: {model_name}")

def get_available_models():
    """
    Returns a list of available model names.
    
    Returns:
        list: List of available model names
    """
    return ['resnet18', 'resnet50', 'vgg16', 'vision_transformer'] 