from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch.nn.functional as F
import numpy as np
import torch

def evaluate_model(model, val_dataloader, device, class_names=["Negative", "Positive"]):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['pad_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    print(f"\n✅ Validation Accuracy: {accuracy:.4f}")
    print("\n📊 Confusion Matrix:")
    print(conf_matrix)
    print("\n📈 Classification Report:")
    print(report)

    return accuracy, conf_matrix, report