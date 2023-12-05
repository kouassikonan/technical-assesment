import torch.nn as nn
from torchvision import models

class ResNet18WithDropout(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(ResNet18WithDropout, self).__init__()
        # Load pre-trained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)

        # Remove the original fully connected layer
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()

        # Add a new fully connected layer with dropout
        self.fc_with_dropout = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc_with_dropout(x)
        return x
    
def create_model():
    return ResNet18WithDropout()

