import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import CC, ACC, PCC, PACC, EMQ

def run_quantification (wrapper, data, val_mask, test_mask, quantifier_model):
    """
    runs the quantification process using the provided wrapper and quantifier model.
    val mask: used to fit the quantifier
    test mask: used to evaluate the quantifier

    """

    all_indices = np.arange(data.num_nodes)
    val_indices = all_indices[val_mask.cpu().numpy()]
    test_indices = all_indices[test_mask.cpu().numpy()]

    val_y = data.y[val_mask.cpu().numpy()]
    test_y = data.y[test_mask.cpu().numpy()]

    validation_set = LabelledCollection(val_indices, val_y.numpy(), classes=wrapper.classes_)
    test_set = LabelledCollection(test_indices, test_y.numpy(), classes=wrapper.classes_)

    true_prev = test_set.prevalence()

    quantifiers_map = {
        'CC': CC(wrapper),
        'ACC': ACC(wrapper),
        'PCC': PCC(wrapper),
        'PACC': PACC(wrapper),
        'EMQ': EMQ(wrapper)
    }

    if quantifier_model == 'all':
        selection = quantifiers_map
    elif quantifiers_map in quantifiers_map:
        selection = {quantifier_model: quantifiers_map[quantifier_model]}
    else:
        selection = quantifiers_map

    for name, quantifier in selection.items():

        quantifier.fit(validation_set, fit_classifier=False)
        estimated_prev = quantifier.quantify(test_set.instances)

        mae = qp.error.mae(true_prev, estimated_prev)
        print(f"{name:<10} | {str(np.round(estimated_prev, 4)):<25} | {mae:.4f}")
