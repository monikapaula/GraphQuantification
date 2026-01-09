import torch
import torch_scatter
import numpy as np
import quapy as qp
from torch_sparse import SparseTensor
from torch_geometric.nn import APPNP

def kronecker_product(*tensors):
    if len(tensors) == 1:
        return tensors[0]

    tensor_count = len(tensors)
    eq_in = []
    out = "..."
    for i in range(tensor_count):
        letter = chr(97 + i)
        eq_in.append(f"...{letter}")
        out += letter
    equation = ",".join(eq_in) + "->" + out
    result = torch.einsum(equation, *tensors)
    return result.reshape(result.shape[:-tensor_count] + (-1,))

def sparse_kronecker_product(*idx_tensors, max_indices):
    dims = len(idx_tensors)
    res = idx_tensors[-1].to(torch.int64)
    factor = 1
    for i in range(1, dims):
        factor *= max_indices[-i]
        res = res + (idx_tensors[-i -1].to(torch.int64) * factor)
    return res


def sparse_kronecker_product_sum(*idx_tensors, max_indices, values=None):
    dims = len(idx_tensors)
    assert dims == len(max_indices)
    idxs = sparse_kronecker_product(*idx_tensors, max_indices= max_indices)
    max_index_prod = int(np.prod(max_indices))

    if values is None:
        values = torch.tensor(1, dtype=torch.int64)
    if values.dim() > idxs.dim():
        idxs = idxs.expand_as(values)
    result = torch.zeros(idxs.shape[:-1] + (max_index_prod,), dtype=values.dtype)
    result.scatter_add_(-1, idxs, values.expand_as(idxs))

    return result

#ACC
def hard_multi_cond_prob_estimate(y_trues, y_preds, num_classes, y_true_weights= None, normalize: bool = True):
    y_true = [t.long() for t in y_trues]
    y_pred = [t.long() for t in y_preds]
    max_indices = [num_classes] * (len(y_trues) + len(y_preds))

    confusion = sparse_kronecker_product_sum(
        *y_true, *y_pred, max_indices=max_indices, values=y_true_weights).to(dtype=torch.float32)

    confusion = confusion.view(confusion.shape[:-1] + (num_classes ** len(y_trues), -1))

    if normalize:
        class_counts = confusion.sum(dim=-1, keepdim=True)
        row_sums = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)

        normalized_confusion = confusion / row_sums

        return normalized_confusion

    return confusion

#PACC
def soft_multi_prob_cond_prob_estimate(y_trues, y_preds_soft, num_classes, y_true_weights=None, normalize=True, EPS: float = 1e-6):
    y_true = y_trues[0].long()
    y_pred_soft = y_preds_soft[0]

    if y_true_weights is not None:
        y_pred_soft= y_pred_soft * y_true_weights.unsqueeze(-1)

    confusion = torch.zeros(num_classes, y_pred_soft.shape[-1], device=y_pred_soft.device)
    torch_scatter.scatter_add(y_pred_soft, y_true, dim=0, out=confusion)

    if normalize:
        class_counts = confusion.sum(dim=-1, keepdim=True)
        confusion = confusion / torch.where(class_counts < EPS, torch.ones_like(class_counts), class_counts)

    return confusion

def soft_multi_prob_estimate(*y_preds):
    joint_pred_dists = kronecker_product(*y_preds)
    return joint_pred_dists.mean(dim=-2)

def compute_weights(
        data,
        val_mask,
        test_mask,
        num_classes,
        depth_limit=10,
        alpha=0.1):
    N = int(data.num_nodes)
    device = data.edge_index.device
    identity = torch.eye(N, dtype=torch.float32, device=device)
    adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(N, N), trust_data=True)
    propagate = APPNP(K=depth_limit, alpha=alpha, add_self_loops=True).to(device)

    weights = propagate(identity, adj_t)

    sis_weights = weights[val_mask][:, test_mask].sum(dim=-1) #1297
    return sis_weights

def compute_confusion(data, val_mask, test_mask, wrapper, num_classes, save_path, mode):
    split_weights = compute_weights(data, val_mask, test_mask, num_classes)
    y_val = data.y[val_mask]
    y_val_pred_probs = torch.tensor(wrapper.predict_proba(val_mask), device=y_val.device)
    if mode == 'acc':
        y_hat_val = y_val_pred_probs.argmax(dim=-1)
        confusion = hard_multi_cond_prob_estimate(
            y_trues=[y_val],
            y_preds = [y_hat_val],
            num_classes=num_classes,
            y_true_weights=split_weights,
        )
    else:
        confusion = soft_multi_prob_cond_prob_estimate(
            y_trues=[y_val],
            y_preds_soft=[y_val_pred_probs],
            num_classes=num_classes,
            y_true_weights=split_weights,
        )

    confusion = confusion.transpose(-1,-2)

    torch.save({
        'sis_confusion_matrix': confusion.cpu(), 'mode':mode
    }, save_path)

def quantification_ppr(save_path, wrapper, test_mask):
    load_conf = torch.load(save_path, weights_only=False)
    confusion = load_conf['sis_confusion_matrix'].numpy()
    mode = load_conf.get('mode', 'pacc')
    num_classes = confusion.shape[0]
    y_test_pred_probs = wrapper.predict_proba(test_mask)
    if mode == 'acc':
        y_hat_test = np.argmax(y_test_pred_probs, axis=1)
        test_cc = np.bincount(y_hat_test, minlength=num_classes)/ len(y_hat_test)
    else:
        test_cc = soft_multi_prob_estimate(torch.tensor(y_test_pred_probs)).numpy()

    adjusted_quant =qp.functional.solve_adjustment(
        class_conditional_rates= confusion,
        unadjusted_counts=test_cc,
        method='inversion',
        solver= 'minimize'
    )
    print('quantification_ppr: mode', mode, 'test_cc', test_cc, 'adjusted_quant', adjusted_quant)
    return adjusted_quant



