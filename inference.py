
import argparse
from PIL import Image
import torch
from torchvision import transforms
from model import create_model
import os

def inference_single(model_path, image_path, class_names):
    model = create_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path)
    img = image_transforms(img).float()
    img = img.unsqueeze(0)

    output = model(img)
    _, predicted = torch.max(output.data, 1)
    
    predicted_class = class_names[predicted.item()]
    print(f"Predicted class: {predicted_class}")

def inference(model_path, image_path=None, folder_path=None, class_names=None):
    if image_path:
        print(f"Inferencing on single image: {image_path}")
        inference_single(model_path, image_path, class_names)
    elif folder_path:
        print(f"Inferencing on images in folder: {folder_path}")
        for dir in os.listdir(folder_path):
            directory = os.path.join(folder_path, dir)
            for file in os.listdir(directory):
                image_path = os.path.join(directory, file)
                print(image_path)
                inference_single(model_path, image_path, class_names)
    else:
        print("Please provide either --image_path or --folder_path for inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image inference using a trained model.")
    parser.add_argument("model_path", type=str, help="Path to the model dict.")
    parser.add_argument("--image_path", type=str, help="Path to the input image.")
    parser.add_argument("--folder_path", type=str, help="Path to the input folder.") 
    args = parser.parse_args()

    # Load your trained model
    model_path = args.model_path
    model = torch.load(model_path)  # Update with your actual model path
    
    # Define your class names
    class_names = ["fields", "roads"]

    # Run inference
    inference(model_path, args.image_path, args.folder_path, class_names)

    # python script_name.py --model_path your_model.pth --image_path path_to_single_image --folder_path path_to_folder