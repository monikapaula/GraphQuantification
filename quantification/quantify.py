import os
import numpy as np
import quapy as qp
import argparse
import torch

from quapy.data import LabelledCollection
from quapy.method.aggregative import CC,ACC, PCC, PACC, KDEyML
from quapy.model_selection import GridSearchQ
from quapy.protocol import APP

from utils.data_loader import load_data_object, load_model
from create_splits.split_manager import load_split
from quantification.wrapper import WrapperClassifier
from utils.sis import compute_confusion, quantification_ppr
from utils.metrics import plot_probability_distribution

def quantify(DATASET_NAME, SPLIT_NAME, CLASSIFIER_MODEL):
    BASE_DIR = 'split_data'
    DEVICE = 'cpu'

    data = load_data_object(DATASET_NAME, base_dir=BASE_DIR, split_name=SPLIT_NAME)
    #print("Shape",data.x.shape)
    num_nodes = data.num_nodes
    #print(f"Dataset '{DATASET_NAME}' loaded. Nodes: {data.num_nodes}, Features: {data.x.size(1)}")
    model, loaded_config = load_model(
        dataset_name=DATASET_NAME,
        model_type=CLASSIFIER_MODEL,
        split_name=SPLIT_NAME,
        model_config= None,
        device=DEVICE,
    )
    assert loaded_config['input_dim'] == data.x.size(1), \
        f"Input dim mismatch: checkpoint={loaded_config['input_dim']} data={data.x.size(1)}"
    #print(f"Model loaded for dataset='{DATASET_NAME}', split='{SPLIT_NAME}', model='{CLASSIFIER_MODEL}'.")

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

    #Experiments
    val_protocol = APP(val_set, n_prevalences=11, repeats=1, sample_size=len(val_set))
    param_grid = {'bandwidth': np.linspace(0.05,0.2, 10)}


    quantifiers = {
        'CC': CC(wrapper,fit_classifier=False),
        'ACC': ACC(wrapper, fit_classifier=False),
        'PCC': PCC(wrapper, fit_classifier=False),
        'PACC': PACC(wrapper, fit_classifier=False),
        'KDEy': GridSearchQ(model= KDEyML(wrapper, fit_classifier=False), param_grid=param_grid,protocol=val_protocol,error='mae', refit=True)
    }
    plot_probability_distribution(wrapper, val_indices, title="Validation Set Probabilities")
    print(f"\nTrue prevalence: {np.round(test_set.prevalence(), 4)}")
    for name, quantifier in quantifiers.items():
        quantifier.fit(val_set.instances, val_set.labels)
        est_prev = quantifier.quantify(test_set.instances)
        mae = qp.error.mae(test_set.prevalence(), est_prev)
        print(f"{name}: Estimated = {np.round(est_prev, 4)}\tMAE = {mae:.4f}")

    conf_dir = "confusion_matrix"
    os.makedirs(conf_dir, exist_ok=True)
    sis_path= os.path.join(conf_dir, f"{DATASET_NAME}_{SPLIT_NAME}_sis.pt")
    if not os.path.exists(sis_path):
        print(f"Creating Confusion matrix {sis_path}")
        compute_confusion(
            data=data,
            val_mask=val_mask,
            test_mask=test_mask,
            wrapper=wrapper,
            num_classes=num_classes,
            save_path=sis_path
        )
    try:
        est_prev_sis = quantification_ppr(sis_path, wrapper, test_mask)
        mae_sis = qp.error.mae(test_set.prevalence(), est_prev_sis)
        print(f"SIS-ACC: Estimated = {np.round(est_prev_sis, 4)}\tMAE = {mae_sis:.4f}")
    except Exception as e:
        print(f"SIS-ACC: Calculation failed ({e})")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    quantify(args.dataset, args.split, args.model)





