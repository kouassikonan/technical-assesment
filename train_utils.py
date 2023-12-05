from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

def train(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, n_epoch):
    # to use tensorboard 
    #log_dir = "/tensorboard"
    #writer = SummaryWriter(log_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    best_f1 = 0.0 # best f1 score
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    training_losses = []
    val_losses = []
    training_accuracy = []
    # training loop
    for epoch in range(n_epoch):
        print("epoch number %d" % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        
        for data in train_dataloader:
            images, labels = data
            images.to(device)
            labels.to(device)
            total += labels.size(0)
            
            optimizer.zero_grad()
            
            output = model(images)
            _, predicted = torch.max(output.data, 1)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()
        
        # get training accuracy and loss by epoch
        epoch_loss= running_loss/len(train_dataloader)
        epoch_acc = 100.00 * running_correct/ total 
        #writer.add_scalar("Loss/train", epoch_loss, epoch)
        
        print("   - Training dataset got %d out of %d images correctly (%.3f%%). Epoch Loss: %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))
        
        test_acc, precision, recall, f1_score, val_loss= evaluate_test(model, test_dataloader, criterion) # accuracy on test set

        scheduler.step(val_loss)

        # save the metrics and loss values for plotting
        accuracy_values.append(test_acc)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)
        training_losses.append(epoch_loss)
        val_losses.append(val_loss)
        training_accuracy.append(epoch_acc)

        if (f1_score > best_f1):
            best_f1 = f1_score # upadate the best accuracy
            torch.save(model.state_dict(), 'best_ckpt.pth')
    
    print("   - Best f1_sccore: %.3f%%" % best_f1)

    print("training complete")
    plot_metrics(np.arange(1, n_epoch + 1), accuracy_values, training_losses, val_losses, precision_values, recall_values, f1_values, training_accuracy)
    


def evaluate_test(model, test_dataloader, criterion):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    val_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(test_dataloader)

    accuracy = 100 * correct / total
    print("   - Test dataset accuracy: %.3f%%" % accuracy)
    precision = 100*precision_score(all_labels, all_preds, average='weighted')
    recall = 100*recall_score(all_labels, all_preds, average='weighted')
    f1 = 100*f1_score(all_labels, all_preds, average='weighted')

    print("   - Precision: %.3f%%, Recall: %.3f%%, F1 Score: %.3f%%" % (precision, recall, f1))

    # Confusion Matrix
    #conf_matrix = confusion_matrix(all_labels, all_preds)
    #print("   - Confusion Matrix:\n", conf_matrix)

    return accuracy, precision, recall, f1, avg_val_loss

def plot_metrics(epochs, accuracy_values, train_loss_values, val_loss_values, precision_values, recall_values, f1_values, training_accuracy):
    plt.figure(figsize=(6, 6))

    # Subplot 1: test_Accuracy and Training accuracy
    #plt.subplot(3, 1, 1)
    plt.plot(epochs, accuracy_values, label='Test accuracy', color='blue')
    plt.plot(epochs, training_accuracy, label='Training accuracy', color='red')
    plt.title('Accuracy and Training accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Subplot 2: Validation Loss and Training Loss
    plt.figure(figsize=(6, 6))
    #plt.subplot(3, 1, 2)
    plt.plot(epochs, val_loss_values, label='Validation Loss', color='green')
    plt.plot(epochs, train_loss_values, label='Training Loss', color='red')
    plt.ylim(0, 5)

# Set y-axis ticks with a step of 0.5
    plt.yticks(np.arange(0, 5.5, 0.5))
    plt.title('Validation Loss and Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Subplot 3: Precision, Recall, and F1 Score
    plt.figure(figsize=(6, 6))
    #plt.subplot(3, 1, 3)
    plt.plot(epochs, precision_values, label='Precision', color='orange')
    plt.plot(epochs, recall_values, label='Recall', color='purple')
    plt.plot(epochs, f1_values, label='F1 Score', color='green')
    plt.title('Precision, Recall, and F1 Score')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()