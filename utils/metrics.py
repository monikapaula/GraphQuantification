import numpy as np
import torch
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score


def prevalence(y, classes):
    return np.array([(y == c).mean() for c in classes])

def classifier_mae(predictions, ground_truth):
    classes = np.unique(ground_truth)
    true_prev = prevalence(ground_truth, classes)
    pred_prev = prevalence(predictions, classes)
    mae = mean_squared_error(true_prev, pred_prev)

    return mae, true_prev, pred_prev

def extensive_evaluate(model, x, edge_index, mask, y):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)

        y_true = y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        print("\n--- Confusion Matrix ---")
        print(f"[[TN : {cm[0][0]}, FP : {cm[0][1]}]")
        print(f" [FN : {cm[1][0]}, TP : {cm[1][1]}]]")

        print("\n--- Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=['Democrat', 'Republican']))

    marco_f1= f1_score(y_true, y_pred, average='macro')
    return marco_f1