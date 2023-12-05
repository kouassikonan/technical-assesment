import os
from torchvision import datasets, transforms
import torch
import torchvision

def count_images_in_directory(directory_path, image_extensions=['.jpg', '.jpeg', '.png']):
    # Get the list of files in the directory
    file_list = os.listdir(directory_path)

    # Filter files with specified image extensions
    image_files = [file for file in file_list if any(file.lower().endswith(ext) for ext in image_extensions)]

    # Get the number of image files
    num_images = len(image_files)

    return num_images

def create_class_weights(fields_path, roads_path):
    fields_samples = count_images_in_directory(fields_path)
    roads_samples = count_images_in_directory(roads_path)
    total_samples = fields_samples + roads_samples
    class_weights = torch.tensor([total_samples / fields_samples, total_samples / roads_samples])
    return class_weights
    # Implementation of create_class_weights function

def create_data_loaders(train_dir, test_dir, batch_size=8):

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
    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle= True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle= True)
    return train_dataloader, test_dataloader