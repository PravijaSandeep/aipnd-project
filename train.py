import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import copy
import argparse
import os

# Define the command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset.')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint.')
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture (vgg13 or resnet18).')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate.')
    parser.add_argument('--hidden_units', type=int, default=256, help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    return parser.parse_args()

# Load data and apply transforms
def load_data(data_dir):
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])          
        }
  
        # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['valid']),
        }

    #  Using the image datasets and the trainforms, define the dataloaders

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=2, pin_memory=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False, num_workers=2, pin_memory=True),
        }

    
    return image_datasets,dataloaders

# function to train the model
def train_model(model, dataloaders,image_datasets, criterion, optimizer, scheduler,device, num_epochs=10, patience=3):
    """
    Train the model with early stopping and learning rate scheduling.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        criterion (torch.nn.Module): The loss function (e.g., NLLLoss).
        optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam) used to update the model parameters.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler that adjusts the learning rate based on validation loss.
        num_epochs (int): The maximum number of epochs to train the model (default: 10).
        patience (int): The number of epochs to wait for improvement before early stopping (default: 3).

    Returns:
        model (torch.nn.Module): The trained model with the best weights (based on validation accuracy).
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Print the current learning rate from the optimizer
        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            print(f"{phase} in epoch {epoch +1} ")
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Adjust learning rate using scheduler after validation
            if phase == 'valid':
                scheduler.step(epoch_loss)  # Pass validation loss to scheduler

            # Deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0
            elif phase == 'valid':
                no_improve_epochs += 1

        print()

        # Early stopping
        if no_improve_epochs >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Save the checkpoint
def save_checkpoint(model, optimizer, save_dir, arch, hidden_units, epochs, learning_rate, class_to_idx):
    checkpoint = {
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    print(f"Model saved to {save_dir}/checkpoint.pth")

# Main function to parse args and execute training
def main():
    args = get_input_args()

    print(torch.cuda.is_available())
    # Set device to GPU if specified and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load data
    image_datasets,dataloaders = load_data(args.data_dir)
    
    # Load model architecture
    if args.arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze pre-trained layers if desired
        for param in model.parameters():
            param.requires_grad = False
    

        model.fc = nn.Sequential(
            nn.Linear(512, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        # Ensure the last layers (newly added layers) have gradients enabled
        for param in model.fc.parameters():
            param.requires_grad = True

    elif args.arch == 'vgg13':
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        # Freeze pre-trained layers if desired
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        # Ensure the last layers (newly added layers) have gradients enabled
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model architecture: {args.arch}")
    
    
    # Set the class_to_idx mapping from the training dataset to the model
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Define the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


    # update criterion  on the correct device
    criterion = criterion.to(device)

    # Move model to the appropriate device
    model = model.to(device)

    # Train the model
    model = train_model(model,  dataloaders,image_datasets, criterion, optimizer,scheduler, device, args.epochs,patience=3)

    # Save the model checkpoint
    save_checkpoint(model, optimizer, args.save_dir, args.arch, args.hidden_units, args.epochs, args.learning_rate, model.class_to_idx)


if __name__ == '__main__':
    main()
