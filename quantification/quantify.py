import os
import numpy as np
import quapy as qp
import argparse
import pandas as pd
import datetime

from quapy.data import LabelledCollection
from quapy.method.aggregative import CC,ACC, PCC, PACC, KDEyML
from quapy.model_selection import GridSearchQ
from quapy.protocol import APP,UPP
from pathlib import Path

from utils.data_loader import load_data_object, load_model
from create_splits.split_manager import load_split
from quantification.wrapper import WrapperClassifier
from utils.sis import compute_confusion, quantification_ppr
from utils.metrics import kl_divergence,plot_probability_distribution
from sklearn.metrics import confusion_matrix

def quantify(DATASET_NAME, SPLIT_NAME, CLASSIFIER_MODEL, run_sis = False):
    BASE_DIR = 'split_data'
    DEVICE = 'cpu'

    data = load_data_object(DATASET_NAME, base_dir=BASE_DIR, split_name=SPLIT_NAME)
    num_nodes = data.num_nodes
    model, loaded_config = load_model(
        dataset_name=DATASET_NAME,
        model_type=CLASSIFIER_MODEL,
        split_name=SPLIT_NAME,
        model_config= None,
        device=DEVICE,
    )
    wrapper = WrapperClassifier(model, data, device=DEVICE)
    wrapper.fit()

    train_mask, val_mask, test_mask = load_split(DATASET_NAME, SPLIT_NAME, num_nodes)

    all_indices = np.arange(num_nodes)
    classes = np.unique(data.y.cpu().numpy())
    num_classes = len(classes)

    val_indices = all_indices[val_mask]
    test_indices = all_indices[test_mask]
    val_y = data.y[val_mask].cpu().numpy()
    test_y = data.y[test_mask].cpu().numpy()

    val_set = LabelledCollection(val_indices, val_y, classes=classes)
    test_set = LabelledCollection(test_indices, test_y, classes=classes)

    print(f"ANALYSIS: {DATASET_NAME} | Split: {SPLIT_NAME}")
    val_preds = wrapper.predict(val_set.instances)
    test_preds = wrapper.predict(test_set.instances)

    cm_val = confusion_matrix(val_set.labels, val_preds)
    cm_test= confusion_matrix(test_set.labels, test_preds)
    print(f"\nConfusion Matrix (Validation):\n{cm_val}")
    print(f"Confusion Matrix (Test):\n{cm_test}")

    M_val = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
    M_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
    matrix_dist = np.linalg.norm(M_val - M_test)
    print(f"\nMatrix Distance (Val vs Test): {matrix_dist:.4f}")

    p_true = test_set.prevalence()
    p_hat = np.bincount(test_preds, minlength=num_classes) / len(test_preds)
    print(f"\nTrue Prevalence p(y):{np.round(p_true, 4)}")
    print(f"Predicted Prev. p(y_hat): {np.round(p_hat, 4)} ")

    M = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
    try:
        #  M.T * p_corr = p_hat
        p_corr = np.linalg.solve(M.T, p_hat)
        p_corr = np.clip(p_corr, 0, 1)
        p_corr /= p_corr.sum()
        print(f"Corrected Prev. (ACC):    {np.round(p_corr, 4)}")

        mae_raw = np.abs(p_true - p_hat).mean()
        mae_corr = np.abs(p_true - p_corr).mean()
        print(f"\nMAE Raw: {mae_raw:.44f}")
        print(f"MAE Corrected: {mae_corr:.4f}")

        if mae_corr > mae_raw:
            print("ACHTUNG: Korrektur war SCHÄDLICH (MAE gestiegen).")
    except np.linalg.LinAlgError:
        print("\nMatrix singulär - Gleichungssystem nicht lösbar.")

    print(f"{'=' * 40}\n")

    val_probs = wrapper.predict_proba(val_set.instances)
    test_probs = wrapper.predict_proba(test_set.instances)
    M_p_val = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        row_mask = (val_y == i)
        if row_mask.any():
            M_p_val[i, :] = val_probs[row_mask].mean(axis=0)
    M_p_test = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        row_mask = (test_y == i)
        if row_mask.any():
            M_p_test[i, :] = test_probs[row_mask].mean(axis=0)
    p_matrix_dist = np.linalg.norm(M_p_val - M_p_test)
    print(f"Probabilistic Matrix Distance: {p_matrix_dist:.4f}")
    p_hat_p = test_probs.mean(axis=0)
    try:
        p_corr_pacc = np.linalg.solve(M_p_val.T, p_hat_p)
        p_corr_pacc = np.clip(p_corr_pacc, 0, 1)
        p_corr_pacc /= p_corr_pacc.sum()

        mae_pacc = np.abs(p_true - p_corr_pacc).mean()
        print(f"Corrected Prev. (PACC):   {np.round(p_corr_pacc, 4)}")
        print(f"MAE PACC: {mae_pacc:.4f}")

        if mae_pacc > mae_raw:
            print("ACHTUNG: PACC Korrektur war ebenfalls SCHÄDLICH.")
    except np.linalg.LinAlgError:
        print("PACC System nicht lösbar.")

    param_grid = {'bandwidth': [0.1,0.15,0.2]}

    if num_classes <= 2:
        val_protocol = APP(val_set, n_prevalences=11, repeats=10, sample_size=len(val_set))
    else:
        val_protocol = UPP(val_set, repeats=20, sample_size=100)

    quantifiers = {
        'CC': CC(wrapper,fit_classifier=False),
        'ACC': ACC(wrapper, fit_classifier=False),
        'PCC': PCC(wrapper, fit_classifier=False),
        'PACC': PACC(wrapper, fit_classifier=False),
        'KDEy': GridSearchQ(model= KDEyML(wrapper, fit_classifier=False), param_grid=param_grid,protocol=val_protocol,error='mae', refit=True)
    }
    run_results = []
    true_prev = test_set.prevalence()

    #plot_probability_distribution(wrapper, val_indices, title="Validation Set Probabilities")
    print(f"\nTrue prevalence: {np.round(test_set.prevalence(), 4)}")

    for name, quantifier in quantifiers.items():
        quantifier.fit(val_set.instances, val_set.labels)
        est_prev = quantifier.quantify(test_set.instances)
        mae = qp.error.mae(test_set.prevalence(), est_prev)
        kl= kl_divergence(true_prev, est_prev)
        #print(f"{name}: Estimated = {np.round(est_prev, 4)}\tMAE = {mae:.4f}\tKL = {kl:.4f}")

        run_results.append({
            'Dataset': DATASET_NAME,
            'Split': SPLIT_NAME,
            'Classifier': CLASSIFIER_MODEL,
            'Method': name,
            'MAE': mae,
            'KL': kl
        })

    if run_sis:
        conf_dir = os.path.join("quantification_results", "confusion_matrix")
        os.makedirs(conf_dir, exist_ok=True)
        for mode in ['acc', 'pacc']:
            name = f"SIS-{mode.upper()}"
            sis_path= os.path.join(conf_dir, f"{DATASET_NAME}_{SPLIT_NAME}_sis_{mode}.pt")
            if not os.path.exists(sis_path):
                print(f"Creating Confusion matrix {sis_path}")
                compute_confusion(
                    data=data,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    wrapper=wrapper,
                    num_classes=num_classes,
                    save_path=sis_path,
                    mode=mode
                )
            try:
                est_prev_sis = quantification_ppr(sis_path, wrapper, test_mask)
                mae_sis = qp.error.mae(true_prev, est_prev_sis)
                kl_sis = kl_divergence(true_prev, est_prev_sis)
                run_results.append({
                    'Dataset': DATASET_NAME,
                    'Split': SPLIT_NAME,
                    'Classifier': CLASSIFIER_MODEL,
                    'Method': name,
                    'MAE': mae_sis,
                    'KL': kl_sis
                })

            except Exception as e:
                print(f"SIS-ACC failed {e}")

    return run_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--splits', type=str, nargs='+', required=True)
    parser.add_argument('--run_sis', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    all_experiments = []
    timestamp= datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    path = Path(__file__).parent / f"results/quantification_results_{timestamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for split in args.splits:
            for model in args.models:
                try:
                    result = quantify(dataset, split, model, run_sis = args.run_sis)
                    all_experiments.extend(result)
                    df_step = pd.DataFrame(result)
                    df_step.to_csv(path, mode='a', index=False, header=not path.exists())

                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"QUANTIFICATION failed {e}")

