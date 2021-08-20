import os
import numpy as np
import pandas as pd
import numpy as np
import json
import time
from utils import dump_json, open_json
import pandas as pd
from multiprocessing import Pool
import argparse

question_map = {}
abcd_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
question_meta, subject_metadata, df, question_meta_1, question_meta_3, = {}, {}, {}, {}, {}


def process_question(infile, outfile):
    data = {}
    with open(infile, 'r') as fp:
        lines = fp.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            words = line.split(',')
            q_id = int(words[0])
            subjects = eval(eval(','.join(words[1:])))
            data[str(q_id)] = {'subjects': subjects}
    return data


def child_map(question_data):
    global subject_metadata
    max_q = 0
    for q_id in question_data:
        subjects = question_data[q_id]['subjects']
        new_subject_map = [
            subject_metadata[str(d)]['new_id'] for d in subjects]
        child_subjects = []
        for d1 in subjects:
            is_ok = True
            for d2 in subjects:
                if d1 == d2:
                    continue
                if d1 in subject_metadata[str(d2)]['parents']:
                    is_ok = False
                    break
            if is_ok:
                child_subjects.append(d1)
        question_data[q_id]['new_sub_map'] = new_subject_map
        child_subject_map = [subject_metadata[str(d)]['new_id']
                             for d in child_subjects]
        question_data[q_id]['child_map'] = child_subject_map
        question_data[q_id]['childs'] = child_subjects
        child_whole_map = []
        for child in child_subjects:
            parent = subject_metadata[str(child)]['parents']
            parent = [d for d in parent if d]
            parent = [subject_metadata[str(d)]['new_id'] for d in parent]
            child_whole_map.append(parent)
        question_data[q_id]['child_whole_map'] = child_whole_map
        max_q = max(len(child_whole_map), max_q)
    print(max_q)
    return question_data


def convert_questions():
    global question_meta_1, question_meta_3
    input_data = 'data/question_metadata_task_1_2.csv'
    output_data = 'data/question_metadata_task_1_2.json'
    if os.path.isfile(output_data):
        question_meta_1 = open_json(output_data)
    else:
        question_meta_1 = child_map(process_question(input_data, output_data))
        dump_json(output_data, question_meta_1)

    input_data = 'data/question_metadata_task_3_4.csv'
    output_data = 'data/question_metadata_task_3_4.json'
    if os.path.isfile(output_data):
        question_meta_3 = open_json(output_data)
    else:
        question_meta_3 = child_map(process_question(input_data, output_data))
        dump_json(output_data, question_meta_3)
    return question_meta_1, question_meta_3


def convert_subjects():
    file_name = 'data/subject_metadata.csv'
    output_data = 'data/subject_metadata.json'
    if os.path.isfile(output_data):
        return open_json(output_data)

    data = {}
    cnt = 1
    with open(file_name, 'r') as fp:
        lines = fp.readlines()[1:]
        lines = [line.strip('\n') for line in lines]
        for line in lines:
            words = line.split(',')
            subject_id = int(words[0])
            if words[-2] == 'NULL':
                parent_id = 0
            else:
                parent_id = int(words[-2])
            level = int(words[-1])
            name = ','.join(words[1:-2])
            data[subject_id] = {'name': name, 'level': level,
                                'parent_id': parent_id, 'parents': [parent_id], 'new_id': cnt}
            cnt += 1
    for subject_id in data:
        while True:
            last_parent = data[subject_id]['parents'][-1]
            if last_parent <= 0:
                break
            data[subject_id]['parents'].append(data[last_parent]['parent_id'])

    dump_json(output_data, data)
    return open_json(output_data)


def f(user_id):
    global df, question_metadata
    user_df = df[df.UserId == user_id].sort_values('DateAnswered')
    q_ids, a_ids, correct_ans, ans, labels, times = [], [], [], [], [], []
    subject_ids = []
    last_timestamp = None
    for _, row in user_df.iterrows():
        q_ids.append(int(row['QuestionId']))
        a_ids.append(int(row['AnswerId']))
        correct_ans.append(int(row['CorrectAnswer']))
        ans.append(int(row['AnswerValue']))
        labels.append(int(row['IsCorrect']))
        if len(times) > 0:
            times.append(float(pd.Timedelta(
                row['DateAnswered'] - last_timestamp).seconds/86400.))
        else:
            times.append(0.)
        last_timestamp = row['DateAnswered']
    subject_ids = [question_metadata[str(d)]['child_map'] for d in q_ids]
    out = {'user_id': int(user_id), 'subject_ids': subject_ids, 'q_ids': q_ids, 'a_ids': a_ids, 'correct_ans': correct_ans,
           'ans': ans, 'labels': labels, 'times': times}
    return out


def featurize(dataset):
    global question_meta_1, question_meta_3, question_metadata, subject_metadata, df
    if dataset == '1_2':
        question_metadata = question_meta_1
    else:
        question_metadata = question_meta_3
    TRAIN_DATA = 'data/train_task_'+dataset+'.csv'
    ANSWER_DATA = 'data/answer_metadata_task_'+dataset+'.csv'

    # AnswerId,DateAnswered,Confidence,GroupId,QuizId,SchemeOfWorkId
    answer_df = pd.read_csv(ANSWER_DATA)[
        ['AnswerId', 'DateAnswered']]
    answer_df['DateAnswered'] = pd.to_datetime(
        answer_df['DateAnswered'], errors='coerce')
    print(answer_df.shape)

    # QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue
    train_df = pd.read_csv(TRAIN_DATA)
    print('train_df shape: ', train_df.shape)
    # print(train_df.isnull().values.any())
    correct_df = train_df[['QuestionId', 'CorrectAnswer']
                          ].drop_duplicates('QuestionId')
    print('correct qs shape: ', correct_df.shape)

    # get answer id info for train
    train_merged_df = pd.merge(train_df, answer_df, on='AnswerId')
    print(train_merged_df.shape)
    print(train_merged_df.isnull().values.any())

    df = train_merged_df

    user_ids = df['UserId'].unique()
    user_data = []
    start_time = time.time()
    with Pool(30) as p:
        user_data = p.map(f, user_ids)
    end_time = time.time()
    print(end_time-start_time)

    print('no of user: ', len(user_data))
    dump_json('data/train_task_'+dataset+'.json', user_data)


def f_ednet(file_name):
    global question_map
    RAW_DIR = '../data/KT1/'
    path = RAW_DIR+file_name
    user_id = file_name.split('.')[0][1:]
    user_df = pd.read_csv(path).sort_values('timestamp')
    q_ids, labels = [], []
    q_ids_set = set()
    for _, row in user_df.iterrows():
        try:
            ans = abcd_map[row['user_answer']]
        except:
            print(user_id)
            print(row)
            continue
        q_id = int(row['question_id'][1:])
        if q_id in q_ids_set:
            continue
        q_ids_set.add(q_id)
        correct_ans = question_map[q_id]['correct_ans']
        ##
        q_ids.append(question_map[q_id]['id'])
        if correct_ans == ans:
            labels.append(1)
        else:
            labels.append(0)
    out = {'user_id': int(user_id), 'q_ids': q_ids,  'labels': labels}
    return out


def featurize_ednet():
    global question_map
    question_df = pd.read_csv('../data/contents/questions.csv')
    for _, row in question_df.iterrows():
        q_id = int(row['question_id'][1:])
        correct_answer = abcd_map[row['correct_answer']]
        question_map[q_id] = {
            'id': len(question_map), 'correct_ans': correct_answer}
    RAW_DIR = '../data/KT1/'
    file_names = os.listdir(RAW_DIR)
    with Pool(30) as p:
        results = p.map(f_ednet, file_names)
    bad_interactions = [len(d['q_ids'])
                        for d in results if len(d['q_ids']) < 20]
    results = [d for d in results if len(d['q_ids']) >= 20]
    interactions = [len(d['q_ids']) for d in results]
    print('Number of Ednet User: ', len(results))
    print('Number of Ednet Interactions: ', sum(interactions))
    print('Ignored Ednet Interactions: ', sum(bad_interactions))
    print('Number of Ednet Problems:', len(question_map))

    dump_json('data/train_task_ednet.json', results)
    dump_json('data/question_map_ednet.json', question_map)


def f_junyi(user_id):
    global df, question_map
    user_df = df[df.uuid == user_id].sort_values('timestamp_TW')
    q_ids, labels = [], []
    q_ids_set = set()
    for _, row in user_df.iterrows():
        q_id = str(row['upid'])
        if q_id in q_ids_set:
            continue
        q_ids_set.add(q_id)
        q_ids.append(question_map[q_id])
        ans = 1 if row['is_correct'] else 0
        labels.append(ans)
    out = {'user_id': user_id, 'q_ids': q_ids, 'labels': labels}
    return out


def featurize_junyi():
    global question_map, df
    df = pd.read_csv('data/junyi/Log_Problem.csv',
                     usecols=['timestamp_TW', 'uuid', 'upid', 'is_correct'])
    print(df.dtypes)
    user_ids = df['uuid'].unique()
    problems = df['upid'].unique()
    for p in problems:
        question_map[str(p)] = len(question_map)
    with Pool(30) as p:
        results = p.map(f_junyi, user_ids)
    bad_interactions = [len(d['q_ids'])
                        for d in results if len(d['q_ids']) < 20]
    results = [d for d in results if len(d['q_ids']) >= 20]
    interactions = [len(d['q_ids']) for d in results]
    print('Number of Junyi User: ', len(results))
    print('Number of Junyi Interactions: ', sum(interactions))
    print('Ignored Junyi Interactions: ', sum(bad_interactions))
    print('Number of Problems Junyi:', len(question_map))

    dump_json('data/train_task_junyi.json', results)
    dump_json('data/question_map_junyi.json', question_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--dataset', type=str,
                        default='assist2009', help='type eedi-1 or eedi-3')
    params = parser.parse_args()
    if params.dataset == 'eedi':
        # Eedi Dataset
        subject_metadata = convert_subjects()
        convert_questions()
        featurize(dataset='3_4')
        featurize(dataset='1_2')
    if params.dataset == 'junyi':
        featurize_junyi()
    if params.dataset == 'ednet':
        featurize_ednet()
