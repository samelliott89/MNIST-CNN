import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
import kagglehub

class MnistDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            train (bool): If True, load training data, else test data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        
        # Adjust path based on dataset structure
        # First check if we have the expected structure
        if self.train:
            self.folder_path = os.path.join(data_dir, 'trainingSet', 'trainingSet')
            if not os.path.exists(self.folder_path):
                # Try alternative path
                self.folder_path = os.path.join(data_dir, 'trainingSet')
                if not os.path.exists(self.folder_path):
                    print(f"Warning: Could not find training data at {self.folder_path}")
        else:
            self.folder_path = os.path.join(data_dir, 'testSet', 'testSet')
            if not os.path.exists(self.folder_path):
                # Try alternative path
                self.folder_path = os.path.join(data_dir, 'testSet')
                if not os.path.exists(self.folder_path):
                    print(f"Warning: Could not find test data at {self.folder_path}")
        
        # Create a list of all file paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        # Verify the folder path exists
        if os.path.exists(self.folder_path):
            # Loop through digit folders (0-9)
            for digit in range(10):
                digit_folder = os.path.join(self.folder_path, str(digit))
                if os.path.exists(digit_folder):
                    # Get all images in this digit folder
                    img_files = [f for f in os.listdir(digit_folder) 
                                if f.endswith('.jpg') or f.endswith('.png')]
                    print(f"Found {len(img_files)} images for digit {digit}")
                    
                    for img_file in img_files:
                        self.image_paths.append(os.path.join(digit_folder, img_file))
                        self.labels.append(digit)
                else:
                    print(f"Warning: Digit folder {digit_folder} not found")
        else:
            print(f"Warning: Folder path {self.folder_path} does not exist")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def download_mnist_dataset():
    """
    Download MNIST dataset using kagglehub
    """
    print("Downloading MNIST dataset from Kaggle...")
    path = kagglehub.dataset_download("scolianni/mnistasjpg")
    print(f"Dataset downloaded to: {path}")
    return path

def get_data_loaders(batch_size=64, download=True):
    """
    Create data loaders for training and testing
    """
    # Download dataset if needed
    if download:
        data_dir = download_mnist_dataset()
    else:
        # Use a local path if dataset is already downloaded
        data_dir = "./mnistasjpg"  # Change this to your local path if needed
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to standard MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create datasets
    train_dataset = MnistDataset(data_dir=data_dir, train=True, transform=transform)
    test_dataset = MnistDataset(data_dir=data_dir, train=False, transform=transform)
    
    # Print dataset sizes for debugging
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    if len(test_dataset) == 0:
        print("WARNING: Test dataset is empty! Checking directory structure...")
        # Print the directory structure to debug
        if os.path.exists(data_dir):
            print(f"Dataset directory exists: {data_dir}")
            # List top-level directories
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    print(f"  Directory: {item}")
                    # List subdirectories
                    if item in ['trainingSet', 'testSet']:
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isdir(subitem_path):
                                print(f"    Subdirectory: {subitem}")
        else:
            print(f"Dataset directory does not exist: {data_dir}")
            
        # If test dataset is empty, use a portion of training dataset as test
        print("Using a portion of training dataset as test dataset instead.")
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
        print(f"New train dataset size: {len(train_dataset)} samples")
        print(f"New test dataset size: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader