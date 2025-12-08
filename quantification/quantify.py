import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import ACC, PCC, PACC, KDEyCS

from utils.data_loader import load_data_object, load_model
from create_splits.split_manager import load_split
from quantification.wrapper import WrapperClassifier
from train_classifier import MODEL_CONFIG

DATASET_NAME = 'deezer_europe'
SPLIT_NAME = 'split_3'
CLASSIFIER_MODEL = 'GCN'
DEVICE = 'cpu'
BASE_DIR = 'split_data'

def quantify(MODEL_CONFIG, DATASET_NAME, SPLIT_NAME, CLASSIFIER_MODEL, DEVICE, BASE_DIR):

    data = load_data_object(DATASET_NAME, base_dir=BASE_DIR)
    print("Shape",data.x.shape)
    num_nodes = data.num_nodes
    #print(f"Dataset '{DATASET_NAME}' loaded. Nodes: {data.num_nodes}, Features: {data.x.size(1)}")
    MODEL_CONFIG['input_dim'] = data.x.size(1)
    MODEL_CONFIG['output_dim'] = len(np.unique(data.y.cpu().numpy()))

    model = load_model(DATASET_NAME, CLASSIFIER_MODEL, SPLIT_NAME, MODEL_CONFIG,DEVICE)
    #print(f"Model loaded for dataset='{DATASET_NAME}', split='{SPLIT_NAME}', model='{CLASSIFIER_MODEL}'.")

    wrapper = WrapperClassifier(model, data, device=DEVICE)
    wrapper.fit()

    train_mask, val_mask, test_mask = load_split(DATASET_NAME, SPLIT_NAME, num_nodes)

    all_indices = np.arange(num_nodes)
    classes = np.unique(data.y.cpu().numpy())

    val_indices = all_indices[val_mask]
    test_indices = all_indices[test_mask]
    val_y = data.y[val_mask].cpu().numpy()
    test_y = data.y[test_mask].cpu().numpy()

    val_set = LabelledCollection(val_indices, val_y, classes=classes)
    test_set = LabelledCollection(test_indices, test_y, classes=classes)

    quantifiers = {
        'ACC': ACC(wrapper, fit_classifier=False),
        'PCC': PCC(wrapper, fit_classifier=False),
        'PACC': PACC(wrapper, fit_classifier=False),
        'KDEy': KDEyCS(wrapper, fit_classifier=False)
    }

    print(f"\nTrue prevalence: {np.round(test_set.prevalence(), 4)}")
    for name, quantifier in quantifiers.items():
        quantifier.fit(val_set.instances, val_set.labels)
        est_prev = quantifier.quantify(test_set.instances)
        mae = qp.error.mae(test_set.prevalence(), est_prev)
        print(f"{name}: Estimated = {np.round(est_prev, 4)}\tMAE = {mae:.4f}")

if __name__ == '__main__':
    quantify(MODEL_CONFIG, DATASET_NAME, SPLIT_NAME, CLASSIFIER_MODEL, DEVICE, BASE_DIR)





