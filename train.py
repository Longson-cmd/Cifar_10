import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_dataloaders
from model import SimpleNeuralNetwork
from utils import save_checkpoint, load_checkpoint, log_confusion_matrix

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_path="checkpoint.pth"):
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, checkpoint_path)
    writer = SummaryWriter()
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        correct_train, total_train, running_loss_train = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Training {epoch+1}/{num_epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            running_loss_train += loss.item()
        
        train_acc = 100 * correct_train / total_train
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        
        model.eval()
        correct_val, total_val, all_labels, all_preds = 0, 0, [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f'Validating {epoch+1}/{num_epochs}', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        val_acc = 100 * correct_val / total_val
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        log_confusion_matrix(all_labels, all_preds, epoch, writer, cifar10_classes)
        
        save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path="checkpoint.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model_best.pth")
            print(f"New best model saved: {best_val_acc:.2f}%")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_loader, test_loader = get_dataloaders()
    model = SimpleNeuralNetwork().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    train(model, train_loader, test_loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), num_epochs=4)

