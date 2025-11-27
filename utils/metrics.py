import numpy as np
import torch
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score
from sympy import print_tree


def prevalence(y, classes):
    return np.array([(y == c).mean() for c in classes])

def classifier_mae(predictions, ground_truth):
    classes = np.unique(ground_truth)
    true_prev = prevalence(ground_truth, classes)
    pred_prev = prevalence(predictions, classes)
    abs_error = np.abs(true_prev - pred_prev)
    mae = np.mean(abs_error)

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
        print(classification_report(y_true, y_pred, target_names=['Democrat', 'Republican'], zero_division=0))

    marco_f1= f1_score(y_true, y_pred, average='macro', zero_division=0)
    return marco_f1

def class_balance(y:torch.Tensor, mask,name):

    if y.dim() != 1:
        y=y.view(-1)
    y_region = y[mask]
    if y_region.numel() == 0:
        print(f"{name} region has no samples.")
        return
    total = y_region.numel()
    dem = (y_region == 0).sum().item()
    rep = (y_region == 1).sum().item()

    print(f"{name} region:")
    print(f"Total samples: {total}")
    print(f"Democrat samples: {dem} ({dem/total:.2%})")
    print(f"Republic samples: {rep} ({rep/total:.2%})")