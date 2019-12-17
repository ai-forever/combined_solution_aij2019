import json
import os
import requests
from utils import read_config
import numpy as np
import tqdm

def get_matching_score(y_true, pred):
    score = 0
    if len(y_true) != len(pred):
        return 0
    for key in y_true.keys():
        if y_true[key] == pred.get(key):
            score += 1
    return score


def get_multiple_score(y_true, pred):
    score = 0
    for y in y_true:
        for p in pred:
            if y == p:
                score += 1
    return score


def get_score(y_true, prediction):
    if y_true == prediction:
        return 1
    return 0


def score_exam(tasks, predictions):
    first_score = 0
    solver2score = dict()
    for task_index, (task, prediction) in enumerate(zip(tasks, predictions), 1):
        task_type = task["question"]["type"]
        if task_type == "matching":
            type_ = "matching"
            y_true = task['solution']['correct']
            score = get_matching_score(y_true, prediction)
        elif task_index == 16:
            type_ = "multiple"
            y_true = task["solution"]["correct_variants"][
                0] if "correct_variants" in task["solution"] \
                else task["solution"]["correct"]
            score = get_multiple_score(y_true, prediction)
        elif task_index == 27:
            type_ = "essay"
            score = 0
        else:
            if "correct_variants" in task["solution"]:
                type_ = "membership accuracy"
                y_true = task["solution"]["correct_variants"]
                score = int(prediction in y_true)
            else:
                type_ = "accuracy"
                y_true = task["solution"]["correct"]
                score = int(prediction == y_true)
        first_score += score
        solver2score[task_index] = score
        print("Score of task {}: {} ({} / {} / {})".format(task_index, score,
                                                      prediction, task["solution"].get("correct")
                                                      or task["solution"].get("correct_variants"), type_))
    return first_score, solver2score


def run_tasks(dir_path):
    tasks, filenames = [], [os.path.join(dir_path, f) for f in
                            os.listdir(dir_path)]
    solver2scores = dict()
    first2second = read_config("./scoring.json")["secondary_score"]
    first_scores, second_scores = list(), list()
    for filename in tqdm.tqdm(filenames):
        if filename.endswith(".json"):
            with open(filename) as f:
                print(filename)
                data = json.load(f)
                resp = requests.post('http://localhost:8000/take_exam',
                                     json={'tasks': data})
                resp_json = resp.json()
                print(resp_json)
                predictions = [resp_json["answers"][str(num)] for num in range(1, 28)]
                first_score, solver2score = score_exam(tasks=data,
                                         predictions=predictions)
                second_score = int(first2second[str(int(first_score))])
                first_scores.append(first_score)
                second_scores.append(second_score)
                for key, value in solver2score.items():
                    if key in solver2scores:
                        solver2scores[key].append(value)
                    else:
                        solver2scores[key] = [value]
                print(
                    "The answers have been scored. First score: {}, second score: {}".format(
                        first_score, second_score))
    print("Mean first score:", np.mean(first_scores))
    print("Max first score:", max(first_scores))
    print("Min first score:", min(first_scores))
    print("")
    print("Mean second score:", np.mean(second_scores))
    print("Max second score:", max(second_scores))
    print("Min second score:", min(second_scores))
    print("")
    print("Mean scores by solver:")
    for solver in sorted(solver2scores.keys()):
        scores = solver2scores[solver]
        print("Solver {}: mean score {}".format(solver, np.mean(scores)))
    return


print(requests.get('http://localhost:8000/ready'))

run_tasks('test_data')
