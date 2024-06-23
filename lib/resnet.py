import torch.hub as hub


def load_resnet(model_name):
    if model_name == "resnet18":
        return hub.load("pytorch/vision", "resnet18", weights="ResNet18_Weights.DEFAULT")
    elif model_name == "resnet34":
        return hub.load("pytorch/vision", "resnet34", weights="ResNet34_Weights.DEFAULT")
    elif model_name == "resnet50":
        return hub.load("pytorch/vision", "resnet50", weights="ResNet50_Weights.DEFAULT")
    elif model_name == "resnet101":
        return hub.load("pytorch/vision", "resnet101", weights="ResNet101_Weights.DEFAULT")
    elif model_name == "resnet152":
        return hub.load("pytorch/vision", "resnet152", weights="ResNet152_Weights.DEFAULT")
    else:
        raise ValueError("Unsupported ResNet architecture. Choose from 'resnet18', 'resnet34', 'resnet50', "
                         "'resnet101', or 'resnet152'.")
