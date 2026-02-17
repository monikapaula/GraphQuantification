import os
import numpy as np
import quapy as qp
import argparse
import pandas as pd
import datetime
import logging

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
from tabulate import tabulate

def quantify(DATASET_NAME, SPLIT_NAME, CLASSIFIER_MODEL, app_logger, run_sis = False):
    BASE_DIR = 'split_data'
    DEVICE = 'cpu'

    header = f"\n{'=' * 60}\n" \
             f"DATASET: {DATASET_NAME} | SPLIT: {SPLIT_NAME} | MODEL: {CLASSIFIER_MODEL}\n" \
             f"{'=' * 60}"
    app_logger.info(header)

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
    classes = np.unique(data.y.cpu().numpy())
    num_classes = len(classes)

    val_y = data.y[val_mask].cpu().numpy()
    test_y = data.y[test_mask].cpu().numpy()
    val_set = LabelledCollection(np.arange(num_nodes)[val_mask], val_y, classes=classes)
    test_set = LabelledCollection(np.arange(num_nodes)[test_mask], test_y, classes=classes)

    val_preds = wrapper.predict(val_set.instances)
    test_preds = wrapper.predict(test_set.instances)
    cm_val = confusion_matrix(val_set.labels, val_preds)
    cm_test = confusion_matrix(test_set.labels, test_preds)

    df_val = pd.DataFrame(cm_val, index=[f"T_{c}" for c in classes], columns=[f"P_{c}" for c in classes])
    df_test = pd.DataFrame(cm_test, index=[f"T_{c}" for c in classes], columns=[f"P_{c}" for c in classes])

    app_logger.info("\n--- VALIDATION CONFUSION MATRIX ---")
    app_logger.info(df_val.to_string())

    app_logger.info("\n--- TEST CONFUSION MATRIX ---")
    app_logger.info(df_test.to_string())

    matrix_dist = matrix_distance(cm_val, cm_test)
    app_logger.info(f"\n --- Matrix Distance --- : {matrix_dist:.5f}")

    M_val = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
    p_true = test_set.prevalence()
    test_preds = wrapper.predict(test_set.instances)
    p_hat = np.bincount(test_preds, minlength=num_classes) / len(test_preds)

    app_logger.info(f"True Prev: {np.round(p_true, 4)}")
    app_logger.info(f"Pred Prev: {np.round(p_hat, 4)}")

    mae_pred = qp.error.mae(p_true, p_hat)
    kl_pred = kl_divergence(p_true, p_hat)
    app_logger.info(f"Prevalence Distances - MAE: {mae_pred:.5f}, KL: {kl_pred:.5f}")

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
    #print(f"\nTrue prevalence: {np.round(test_set.prevalence(), 4)}")

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

    table_headers = ["Method", "MAE", "KL"]
    table_rows = [[r['Method'], f"{r['MAE']:.5f}", f"{r['KL']:.5f}"] for r in run_results]
    summary_table = tabulate(table_rows, headers=table_headers, tablefmt="grid")
    app_logger.info(f"\nFinal Results:\n{summary_table}")

    return run_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--splits', type=str, nargs='+', required=True)
    parser.add_argument('--run_sis', action='store_true')
    return parser.parse_args()


def setup_logger(timestamp):
    log_dir = Path("quantification/results")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_{timestamp}.log"

    l = logging.getLogger("GraphQuant")
    l.setLevel(logging.DEBUG)
    l.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(message)s'))
    l.addHandler(fh)

    return l

def matrix_distance(cm_val, cm_test):
    v_norm = cm_val.astype('float') / (cm_val.sum(axis=1)[:, np.newaxis] + 1e-9)
    t_norm = cm_test.astype('float') / (cm_test.sum(axis=1)[:, np.newaxis] + 1e-9)

    distance = np.linalg.norm(v_norm - t_norm)
    return distance

if __name__ == '__main__':
    args = parse_args()
    all_experiments = []
    timestamp= datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    my_app_logger = setup_logger(timestamp)
    path = Path(__file__).parent / f"results/quantification_results_{timestamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for split in args.splits:
            for model in args.models:
                try:
                    result = quantify(dataset, split, model,my_app_logger, run_sis = args.run_sis)
                    all_experiments.extend(result)
                    df_step = pd.DataFrame(result)
                    df_step.to_csv(path, mode='a', index=False, header=not path.exists())

                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"QUANTIFICATION failed {e}")

