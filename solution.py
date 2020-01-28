import random
from collections import defaultdict
from flask import Flask, request, jsonify
import numpy as np
from utils import *
from solvers import *
from download_models import download_aij_models
import os
import traceback


class CuttingEdgeStrongGeneralAI(object):
    def __init__(self, train_path="public_set/train"):
        self.train_path = train_path
        self.classifier = classifier.Solver()
        if not os.path.exists("./data"):
            download_aij_models("http://bit.ly/2ORHVVC", "aij_data_models.zip")
        solver_classes = [
            solver1,
            solver2,
            solver3,
            solver4,
            solver5,
            solver6,
            solver7,
            solver8,
            solver9,
            solver10,
            solver11,
            solver11,
            solver13,
            solver14,
            solver15,
            solver16,
            solver17,
            solver17,
            solver17,
            solver17,
            solver21,
            solver22,
            solver23,
            solver24,
            solver25,
            solver26,
            solver27
        ]
        self.solvers = self.solver_loading(solver_classes)
        self.clf_fitting()

    def solver_loading(self, solver_classes):
        solvers = []
        for i, solver_class in enumerate(solver_classes):
            solver_index = i + 1
            train_tasks = load_tasks(self.train_path, task_num=solver_index)
            solver = solver_class.Solver()
            try:
                solver.load()
                if not solver.is_loaded:
                    solver.fit(train_tasks)
                    solver.save()
            except Exception as e:
                print("Exception during fitting Solver {}".format(solver_index))
                print(traceback.format_exc())
            print("Solver {} is ready!\n".format(solver_index))
            solvers.append(solver)
        return solvers

    def clf_fitting(self):
        tasks = []
        for filename in os.listdir(self.train_path):
            if filename.endswith(".json"):
                data = read_config(os.path.join(self.train_path, filename))
                tasks.append(data)
        print("Fitting Classifier...")
        self.classifier.fit(tasks)
        print("Classifier is ready!")

    def not_so_strong_task_solver(self, task):
        print("not_so_strong_task_solver!!!", task["id"])
        question = task["question"]
        if question["type"] == "choice":
            choice = random.choice(question["choices"])
            answer = choice["id"]
        elif question["type"] == "multiple_choice":
            min_choices = question.get("min_choices", 1)
            max_choices = question.get("max_choices", len(question["choices"]))
            n_choices = random.randint(min_choices, max_choices)
            random.shuffle(question["choices"])
            answer = [choice["id"] for choice in question["choices"][:n_choices]]
        elif question["type"] == "matching":
            random.shuffle(question["choices"])
            answer = {
                left["id"]: choice["id"]
                for left, choice in zip(question["left"], question["choices"])
            }
        elif question["type"] == "text":
            if question.get("restriction") == "word":
                words = [word for word in task["text"].split() if len(word) > 1]
                answer = random.choice(words)
            else:
                answer = (
                    "Для современного мира реализация намеченных плановых заданий позволяет "
                    "выполнить важные задания по разработке новых принципов формирования "
                    "материально-технической и кадровой базы. Господа, реализация намеченных "
                    "плановых заданий играет определяющее значение для модели развития. "
                    "Сложно сказать, почему сделанные на базе интернет-аналитики выводы призывают "
                    "нас к новым свершениям, которые, в свою очередь, должны быть в равной степени "
                    "предоставлены сами себе. Ясность нашей позиции очевидна: базовый вектор "
                    "развития однозначно фиксирует необходимость существующих финансовых и "
                    "административных условий."
                )
        else:
            raise RuntimeError("Unknown question type: {}".format(question["type"]))

        return answer

    def take_essay(self, essay_task):
        solver = solver27.Solver()
        task = {"text": essay_task["essay"]}
        try:
            ai_essay = solver.predict_from_model(task)
        except Exception as e:
            print(traceback.format_exc())
            ai_essay = self.not_so_strong_task_solver(task)
        return ai_essay

    def take_solver(self, solver_task):
        """Solver task should contain solver id!"""
        data = json.loads(solver_task["task"])[0]
        solv = str(data["model"]).split('.')[1]
        mysolver = eval(solv)
        the_task = data["fields"]
        solver = mysolver.Solver()
        try:
            some_answer = solver.predict_from_model(the_task)
            print("Prediction: ", some_answer)
        except Exception as e:
            print(traceback.format_exc())
            some_answer = self.not_so_strong_task_solver(the_task)
        return some_answer

    def take_exam(self, exam):
        answers = {}
        if "tasks" in exam:
            variant = exam["tasks"]
            if isinstance(variant, dict):
                if "tasks" in variant.keys():
                    variant = variant["tasks"]
        else:
            variant = exam
        task_number = self.classifier.predict(variant)
        print("Classifier results: ", task_number)
        for i, task in enumerate(variant):
            task_id = task["id"]
            try:
                prediction = self.solvers[task_number[i] - 1].predict_from_model(task)
                print("Prediction: ", prediction)
            except Exception as e:
                print(traceback.format_exc())
                prediction = self.not_so_strong_task_solver(task)
            if isinstance(prediction, np.ndarray):
                prediction = list(prediction)
            if isinstance(prediction, str):
                prediction = prediction.replace("ё", "е")
            answers[task_id] = prediction
        return answers


app = Flask(__name__)

ai = CuttingEdgeStrongGeneralAI()


@app.route("/ready")
def http_ready():
    return "OK"


@app.route("/take_essay", methods=["POST"])
def http_take_essay():
    """Return the string of answer"""
    request_data = request.get_json()
    answer = ai.take_essay(request_data)
    return jsonify({"answers": answer})


@app.route("/take_solver", methods=["POST"])
def http_take_solver():
    """Return the string of answer"""
    request_data = request.get_json()
    answers = ai.take_solver(request_data)
    return jsonify({"answers": answers})


@app.route("/take_exam", methods=["POST"])
def http_take_exam():
    request_data = request.get_json()
    answers = ai.take_exam(request_data)
    return jsonify({"answers": answers})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
