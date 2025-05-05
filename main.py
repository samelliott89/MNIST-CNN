import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import MnistCNN
from data_loader import get_data_loaders
from train import train, test, visualize_results, save_checkpoint

# Hyperparameters
batch_size = 128
learning_rate = 0.001
weight_decay = 1e-5
num_epochs = 10  # Train longer but use early stopping

# Early stopping parameters
best_accuracy = 0
patience = 4
patience_counter = 0

def main():
    # Check for available hardware acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Initialize model
    model = MnistCNN().to(device)

    # Define optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1)
    
    # Create data loaders using kagglehub to download the dataset
    train_loader, test_loader = get_data_loaders(batch_size=batch_size, download=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, accuracy = test(model, device, test_loader, criterion)
        
        # Append metrics to lists for plotting
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)
    
        # Update learning rate based on validation loss
        scheduler.step(test_loss)
        
        # Early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, accuracy, 'best_mnist_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break   
                
    # Save final model
    save_checkpoint(model, optimizer, num_epochs, accuracy, 'final_mnist_model.pth')
    
    # Visualize some predictions
    visualize_results(model, device, test_loader)
    
    # Plot training and testing loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs. Epoch')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == '__main__':
    main()