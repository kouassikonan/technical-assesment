from Data_utils import  create_class_weights, create_data_loaders
from model import create_model
from train_utils import train, evaluate_test
import torch.nn as nn
import torch.optim as optim


# Set directory paths
train_dir = "C:/Users/gmkko/Documents/dataset/train_images/"
test_dir = "C:/Users/gmkko/Documents/dataset/test_images/"

# Set classes paths
fields_path = "C:/Users/gmkko/Documents/dataset/train_images/fields"
roads_path = "C:/Users/gmkko/Documents/dataset/train_images/roads"

num_classes = 2
model = create_model()
class_weights = create_class_weights(fields_path, roads_path)

# Define your transformations and create data loaders
train_dataloader,test_dataloader  = create_data_loaders(train_dir, test_dir, batch_size=8)


# Define your loss function and optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.05)
#scheduler = optim.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
# Train your model
trained_model = train(model, train_dataloader, test_dataloader, loss_fn, optimizer,scheduler, 30)