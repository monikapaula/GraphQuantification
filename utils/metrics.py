import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score

def compute_class_weights(y):
    """
    Compute class weights to handle class imbalance
    total_samples / (num_classes * num_samples_per_class)
    """
    classes, counts = y.unique(return_counts=True)
    total_samples = y.size(0)
    num_classes = len(classes)

    weights = total_samples/ (num_classes * counts.float())

    return weights

def prevalence(y, classes):
    return np.array([(y == c).mean() for c in classes])

def classifier_mae(predictions, ground_truth):
    classes = np.unique(ground_truth)
    true_prev = prevalence(ground_truth, classes)
    pred_prev = prevalence(predictions, classes)
    abs_error = np.abs(true_prev - pred_prev)
    mae = np.mean(abs_error)

    return mae, true_prev, pred_prev

def macro_f1(model, x, edge_index, mask, y):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        y_true = y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

    return f1_score(y_true, y_pred, average='macro', zero_division=0)

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
        print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], zero_division=0))

    marco_f1= macro_f1(model, x, edge_index, mask, y)
    return marco_f1

def class_balance(y:torch.Tensor, mask,name):

    if y.dim() != 1:
        y=y.view(-1)
    y_subset = y[mask]
    total = y_subset.numel()

    if y_subset.numel() == 0:
        print(f"{name} has no samples.")
        return

    print(f"---Class balance for {name} set---")
    print(f"Total samples: {total}")

    unique_classes, counts = torch.unique(y_subset, return_counts=True)

    for cls, count in zip(unique_classes, counts):
        cls_id = cls.item()
        count_val = count.item()
        percentage = count_val / total

        print(f"Class {cls_id}: Count = {count_val}, Percentage = {percentage:.4%}")

    print("-" *30)

def plot_probability_distribution(wrapper, indices, title="Probability Distribution"):
    probas = wrapper.predict_proba(indices)[:, 1]

    plt.figure(figsize=(8, 5))
    plt.hist(probas, bins=50, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Predicted Probability (Class 1)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()