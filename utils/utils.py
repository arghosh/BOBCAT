import json
import numpy as np
from sklearn import metrics
import torch
import os
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def list_to_string(socres):
    str_scores = [str(s) for s in socres]
    return ','.join(str_scores)


def data_split(datapath, fold, seed):
    data = open_json(datapath)
    random.Random(seed).shuffle(data)
    fields = ['q_ids',  'labels']  # 'ans', 'correct_ans',
    del_fields = []
    for f in data[0]:
        if f not in fields:
            del_fields.append(f)
    for d in data:
        for f in fields:
            d[f] = np.array(d[f])
        for f in del_fields:
            if f not in fields:
                del d[f]
    N = len(data)//5
    test_fold, valid_fold = fold-1, fold % 5
    test_data = data[test_fold*N: (test_fold+1)*N]
    valid_data = data[valid_fold*N: (valid_fold+1)*N]
    train_indices = [idx for idx in range(len(data))]
    train_indices = [idx for idx in train_indices if idx //
                     N != test_fold and idx//N != valid_fold]
    train_data = [data[idx] for idx in train_indices]

    return train_data, valid_data, test_data


def batch_accuracy(output, batch):
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    target = batch['output_labels'].float().numpy()
    mask = batch['output_mask'].numpy() == 1
    accuracy = torch.from_numpy(np.sum((target == output) * mask, axis=-1) /
                                np.sum(mask, axis=-1)).float()  # B,
    return accuracy


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass
