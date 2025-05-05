import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import MnistCNN
from data_loader import get_data_loaders

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')
    
    # Return average loss for the epoch
    return running_loss / len(train_loader)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += criterion(output, target).item()
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy

def visualize_results(model, device, test_loader, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 8))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    return
                
                images_so_far += 1
                plt.subplot(2, 3, images_so_far)
                plt.tight_layout()
                plt.imshow(inputs.cpu()[j][0], cmap='gray')
                plt.title(f'Predicted: {preds[j].item()}\nActual: {labels[j].item()}')
                plt.xticks([])
                plt.yticks([])
                
    plt.savefig('mnist_predictions.png')
    plt.show()

def save_checkpoint(model, optimizer, epoch, accuracy, filename='mnist_model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")