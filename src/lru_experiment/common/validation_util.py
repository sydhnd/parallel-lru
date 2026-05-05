import torch

def evaluate_model(model, device, data_loader, criterion,threshold=0.0):
    
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.float().view(-1, 1).to(device)
            
            preds = model(features)
            
            if preds.shape != labels.shape:
                preds = preds.view_as(labels)
                
            loss = criterion(preds, labels).to(device)
            total_loss += loss.item()

            pred_labels = (preds > threshold).float()

            total_samples += labels.size(0)
            correct += (pred_labels == labels).sum().item()

            #confusion matrix
            tp += ((pred_labels == 1) & (labels == 1)).sum().item()
            tn += ((pred_labels == 0) & (labels == 0)).sum().item()
            fp += ((pred_labels == 1) & (labels == 0)).sum().item()
            fn += ((pred_labels == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total_samples
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | F1: {f1_score:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"Confusion Matrix -> TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    model.train()
    
    return avg_loss, accuracy, precision, recall, f1_score