import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (    
    ResNet50_Weights,
    ResNet18_Weights,
)

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ConvModel2(nn.Module):
    def __init__(self, num_hidden_layers=2, hidden_size=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        # Create additional hidden layers based on num_hidden_layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc3(x)
        return x
    

def get_conv_model(device):
    return ConvModel().to(device)

def get_conv_model2(device, num_hidden_layers=2, hidden_size=100):
    return ConvModel2(num_hidden_layers, hidden_size).to(device)


def get_resnet_model(model_name, num_classes=10, pretrained=False, device=None):
    """
    Returns a ResNet model (18 or 50)
    
    Args:
        model_name: 'resnet18' or 'resnet50'
        num_classes: number of output classes
        pretrained: whether to use pretrained weights
        device: device to load the model on
    """
    import torchvision.models as models
    
    if model_name == 'resnet18':
        # model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model = models.resnet18(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet50':
        # model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model = models.resnet50(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose 'resnet18' or 'resnet50'")
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    if device is not None:
        model = model.to(device)
    
    return model

def get_resnet18(num_classes=10, pretrained=False, device=None):
    """Returns a ResNet18 model"""
    return get_resnet_model('resnet18', num_classes, pretrained, device)

def get_resnet50(num_classes=10, pretrained=False, device=None):
    """Returns a ResNet50 model"""
    return get_resnet_model('resnet50', num_classes, pretrained, device)

def get_mlp_model(in_features, out_features, hidden_size, device):
    return MLPModel(in_features, out_features, hidden_size).to(device)

def get_model(in_features, out_features, device):
    return LinearModel(in_features, out_features).to(device)