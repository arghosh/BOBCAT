from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from scipy.optimize import brentq
from collections import Counter, defaultdict
import torch
import os
from dataset import Dataset, collate_fn
from utils.utils import open_json, dump_json, compute_auc, compute_accuracy, data_split, batch_accuracy, list_to_string
from utils.configuration import create_parser, initialize_seeds
import time
#import wandb
import neptune
import os
#running in cluster
DEBUG = False  # if torch.cuda.is_available() else True
# if DEBUG:
#os.environ["WANDB_MODE"] = "dryrun"


def sigmoid(x):
    z = np.exp(-x)
    return 1 / (1 + z)


def proba(th, d, bias=0):
    return sigmoid(th - d + bias)


def deriv_likelihood(theta, results):
    return sum(a - proba(theta, d) for d, a in results) - 2*params.policy_lr * (theta-avg_theta)


def estimated_theta(results):
    try:
        return brentq(lambda theta: deriv_likelihood(theta, results), min_theta-1, max_theta+1)
    except ValueError:
        if all(outcome == 1 for _, outcome in results):
            return max_theta
        if all(outcome == 0 for _, outcome in results):
            return min_theta
        return avg_theta


def convert_to_irt(data):
    ds, rows, cols = [], [], []
    n_rows = 0
    Ys = []
    for d in data:
        Ys.append(d['labels'])
        n_item = len(d['labels'])
        row = np.arange(n_item) + n_rows
        col1 = d['q_ids']
        col2 = np.zeros(n_item)+d['user_id']+n_question
        rows.append(row)
        rows.append(row)
        cols.append(col1)
        cols.append(col2)
        n_rows += n_item
        ds.append(np.ones(n_item*2))
    ds, rows, cols = np.concatenate(
        ds), np.concatenate(rows), np.concatenate(cols)
    Xs = csr_matrix((ds, (rows, cols)), shape=(n_rows, n_users+n_question))
    Ys = np.concatenate(Ys)
    return Xs, Ys


def test_model(id_, split='val'):
    if split == 'val':
        valid_dataset.seed = id_
    elif split == 'test':
        test_dataset.seed = id_
    loader = torch.utils.data.DataLoader(
        valid_dataset if split == 'val' else test_dataset, collate_fn=collate_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    total_loss, all_preds, all_targets = 0., [], []
    n_batch = 0
    for batch in loader:
        output = model.forward(batch)
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1
    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return total_loss/n_batch, auc, accuracy


class Model:
    def __init__(self, n_query, n_question, sampling):
        self.n_query = n_query
        self.n_question = n_question
        self.sampling = sampling

    def forward(self, batch):
        input_labels = batch['input_labels'].numpy()
        nb_students, _ = input_labels.shape
        student_thetas = np.array([avg_theta] * nb_students).reshape(-1, 1)
        results = defaultdict(list)
        if self.sampling == 'active':
            input_mask = batch['input_mask'].numpy()
            for _ in range(self.n_query):
                pr = proba(student_thetas, difficulty)
                loss = np.abs(0.5 - pr) + (1 - input_mask)
                selections = np.argmin(loss, axis=1)
                input_mask[range(nb_students), selections] = 0
                #
                for user_id, item_id in enumerate(selections):
                    results[user_id].append(
                        (difficulty[item_id], input_labels[user_id, item_id]))
                for user_id in range(nb_students):
                    student_thetas[user_id] = estimated_theta(results[user_id])
        elif self.sampling == 'random':
            input_mask = batch['input_mask']
            if self.n_query == -1:
                train_mask = input_mask.numpy()
            else:
                train_mask = torch.zeros(input_mask.shape[0], self.n_question)
                selections = torch.multinomial(
                    input_mask.float(), self.n_query, replacement=False)
                train_mask = train_mask.scatter(
                    dim=1, index=selections, value=1)
                train_mask = train_mask.numpy()
            user_ids, item_ids = np.where(train_mask > 0)
            for user_id, item_id in zip(user_ids, item_ids):
                results[user_id].append(
                    (difficulty[item_id], input_labels[user_id, item_id]))
            for user_id in range(nb_students):
                student_thetas[user_id] = estimated_theta(results[user_id])
        # Test
        output = proba(student_thetas, difficulty)
        return output


if __name__ == "__main__":
    params = create_parser()
    print(params)
    project = "arighosh/bobcat"
    initialize_seeds(params.seed)
    #
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    train_data, valid_data, test_data = data_split(
        data_path, params.fold, params.seed)
    valid_dataset, test_dataset = Dataset(valid_data), Dataset(test_data)
    n_users, n_question = len(train_data), params.n_question
    for idx, d in enumerate(train_data):
        d['user_id'] = idx
    Xs, Ys = convert_to_irt(train_data)

    # Train
    lr = LogisticRegression(solver='lbfgs', C=1./params.lr,
                            max_iter=1000, fit_intercept=False)
    lr.fit(Xs, Ys)
    weights = lr.coef_[0]
    thetas = weights[n_question:]
    difficulty = -weights[:n_question]
    avg_theta = np.mean(thetas)
    min_theta = np.min(thetas)
    max_theta = np.max(thetas)
    # Train Ends
    if params.save:
        data = {'avg_theta': float(avg_theta), 'min_theta': float(
            min_theta), 'max_theta': max_theta, 'difficulty': difficulty.tolist()}
        dump_json('model/'+params.file_name, data)
        import sys
        sys.exit()

    #
    sampling = params.model.split('-')[-1]
    model = Model(n_query=params.n_query,
                  n_question=params.n_question, sampling=sampling)
    num_workers = 2
    collate_fn = collate_fn(params.n_question)
    N = [idx for idx in range(100, 100+params.repeat)]
    #
    best_val_score, best_test_score = 0, 0
    best_val_auc, best_test_auc = 0, 0

    # for policy_lr in []:
    for policy_lr in [1e1, 1,  1e-1, 1e-2, 1e-4, 0]:
        params.policy_lr = policy_lr
        # Validation
        val_scores, val_aucs = [], []
        test_scores, test_aucs = [], []
        for idx in N:
            _, auc, acc = test_model(id_=idx, split='val')
            val_scores.append(acc)
            val_aucs.append(auc)
        val_score = sum(val_scores)/(len(N)+1e-20)
        val_auc = sum(val_aucs)/(len(N)+1e-20)
        print('validation', val_score, policy_lr)
        if best_val_score < val_score:
            best_policy_lr = policy_lr
            best_val_score = val_score
            best_val_auc = val_auc
            # Run on test set
            for idx in N:
                _, auc, acc = test_model(id_=idx, split='test')
                test_scores.append(acc)
                test_aucs.append(auc)
            best_test_score = sum(test_scores)/(len(N)+1e-20)
            best_test_auc = sum(test_aucs)/(len(N)+1e-20)
            print('Testing', best_test_score, policy_lr)

    print('Best Test score: {}, Best Valid Score:{}, best policy lr: {}'.format(
        best_test_score, best_val_score, best_policy_lr))
    if not DEBUG:
        params.policy_lr = best_policy_lr
        neptune.init(project_qualified_name=project,
                     api_token=os.environ["NEPTUNE_API_TOKEN"])
        neptune_exp = neptune.create_experiment(
            name=params.file_name, params=vars(params), send_hardware_metrics=False)

        neptune.log_metric('Best Test Accuracy', best_test_score)
        neptune.log_metric('Best Test Auc', best_test_auc)
        neptune.log_metric('Best Valid Accuracy', best_val_score)
        neptune.log_metric('Best Valid Auc', best_val_auc)
