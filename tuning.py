# This function allows the hyperparameter tuning


import optuna
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.metrics import f1_score
import torch, torchvision
from torchvision import transforms
from torch import nn
import torch.optim as optim
from Data_utils import create_class_weights
from model import ResNet18WithDropout
import torch.nn as nn
import torch.optim as optim


# images paths
train_dir = "C:/Users/gmkko/Documents/dataset/train_images/"
test_dir = "C:/Users/gmkko/Documents/dataset/test_images/"
#image classes path
fields_path = "C:/Users/gmkko/Documents/dataset/train_images/fields"
roads_path = "C:/Users/gmkko/Documents/dataset/train_images/roads"


# get the f1_score
def get_f1_score(model, test_dataloader, criterion):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    all_labels = []
    all_preds = []
    # 
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    f1 = 100*f1_score(all_labels, all_preds, average='weighted')



    return  f1
# Adaptation of training function
def optuna_train(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, n_epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    best_f1 = 0.0 # best accuracy on test set

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        
        for data in train_dataloader:
            images, labels = data
            images.to(device)
            labels.to(device)
            total += labels.size(0) # total amount of images
            
            optimizer.zero_grad() # to zero so all the paremeters update correctly
            
            output = model(images)
            _, predicted = torch.max(output.data, 1)

            loss = criterion(output, labels) # loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()
        
        
        f1_score= get_f1_score(model, test_dataloader, criterion) # get f1_score
        if scheduler != None:
            scheduler.step()

        if (f1_score > best_f1):
            best_f1 = f1_score # upadate the best accuracy
    
    print("training complete")

# create a function that try all the hyperparameters given to train the model
def objective(trial):
    # Hyperparameter search space
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int('epoch_num', 10, 50)
    batch_size = trial.suggest_categorical('batch_size', [2,4,8,32,64,128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    learning_rate_scheduler = trial.suggest_categorical('lr_scheduler', [None, 'step', 'cosine'])
    # Create the model
    num_classes  = 2
    model = ResNet18WithDropout(num_classes=num_classes, dropout_rate=0.5)
    class_weights = create_class_weights(fields_path, roads_path)
    loss_fn = nn.CrossEntropyLoss(weight = class_weights)
    # Select optimizer based on the trial suggestion
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Select the scheduler
    # Create learning rate scheduler based on the trial suggestion
    if learning_rate_scheduler == 'step':
        # StepLR scheduler
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust step_size and gamma as needed
    elif learning_rate_scheduler == 'cosine':
        # CosineAnnealingLR scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  # Adjust T_max as needed
    else:
        # No scheduler
        scheduler = scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    training_set = torchvision.datasets.ImageFolder(root= train_dir, transform= train_transforms)
    test_set = torchvision.datasets.ImageFolder(root= test_dir, transform= test_transforms)
    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size, shuffle= True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size, shuffle= True)
    # Set up criterion (replace this with your actual loss function)

    # Training loop
    optuna_train(model, train_dataloader, test_dataloader, loss_fn, optimizer,scheduler, num_epochs)

    # Evaluation
    f1= get_f1_score(model, test_dataloader, loss_fn)

    return f1

# Set up Optuna study, the objective of the study will be to prioritize the max f1_score
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=10), storage="sqlite:///db.sqlite3", study_name="Road_classifier_tunning")  # We want to maximize F1 score
study.optimize(objective, n_trials=10)

# Print the best hyperparameters
print("Best trial:")
trial = study.best_trial
print("Value (F1 Score): ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")