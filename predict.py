import torch
from torchvision import models
from torch import nn
import argparse
import json
from PIL import Image
import numpy as np
from torchvision import transforms

# Command-line argument parser
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability of that name.")
    
    parser.add_argument('input', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')
    
    return parser.parse_args()

# Load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, weights_only=True)
    if checkpoint['arch'] == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Linear(512, checkpoint['hidden_units']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint['hidden_units'], 102),
            nn.LogSoftmax(dim=1)
        )
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, checkpoint['hidden_units']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint['hidden_units'], 102),
            nn.LogSoftmax(dim=1)
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Process an image
def process_image(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array.'''
    # Open the image
    image = Image.open(image_path).convert("RGB")  # Ensure the image has 3 channels (RGB)

    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize shortest side to 256
        transforms.CenterCrop(224),  # Crop to 224x224 in the center
        transforms.ToTensor(),  # Convert the image to a Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Apply the transformations
    image = preprocess(image)

    return image.numpy()  # Convert to numpy array for processing


# Predict the top K classes
def predict(image_path, model, topk=5):
    '''Predict the top K classes of an image using a trained deep learning model.'''
    # Process the image for the model
    image = process_image(image_path)

    # Convert to Tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension

    # Set model  in evaluation mode
    model.eval()
    #move the image to the appropriate device (GPU/CPU)
    image_tensor = image_tensor.to(next(model.parameters()).device)

    # Disable gradient calculation for inference
    with torch.no_grad():
        output = model(image_tensor)

    # Get the probabilities by applying softmax to the model output
    probs = torch.exp(output)

    # Get the top K probabilities and classes
    top_probs, top_classes = probs.topk(topk, dim=1)

    # Convert to numpy arrays for easy use
    top_probs = top_probs.cpu().numpy().squeeze()
    top_classes = top_classes.cpu().numpy().squeeze()

    # Get the class-to-index mapping and reverse it to map indices back to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_classes]

    return top_probs, top_classes


# Main function to handle input and output
def main():
    args = get_input_args()
    
    # Set device (GPU if available and specified, otherwise CPU)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    
    # Predict the top K classes
    top_probs, top_classes = predict(args.input, model,  args.top_k)
    
    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class_names = [cat_to_name[str(cls)] for cls in top_classes]
    else:
        top_class_names = top_classes
    
    # Print the results
    print("Predicted Flower Names and Probabilities:")
    for i in range(args.top_k):
        print(f"{i+1}: {top_class_names[i]} with probability {top_probs[i]:.4f}")

if __name__ == '__main__':
    main()
