# AI Journey 2019: Combined Solution
Русская версия этого документа находится [здесь](README.ru.md).

This is a combined solution of AI Journey 2019 challenge. It consists of refactored code of top-20 solutions from the challenge. Its score is 69.

## Knowledge base and models

Knowledge base of AI Journey 2019 contains data and models, which could be useful for AGI and applied NLP tasks:

* Unified State Exam solving;
* text summarization;
* text generation;
* style transfer;
* punctuation restoring;
* grammar error correction;
* domain-specific language modeling;
* discourse analysis;
* topic modeling;
* text classification.
    
To download the knowledge base please use:

```
python download_data.py
```

Directory ```models``` contains models and additional files for solvers of exam tasks.

To download models please use:

```
python download_models.py
```

## Running a docker container

You can run a container with:
```
$ sudo docker run -w /workspace -v $(pwd):/workspace -p 8000:8000 -it alenush25/combined_solution_aij:latest python solution.py
```

It will run the container with HTTP-server on port `8000`. It supports the following requests:

#### `GET /ready`

The return code will be `200 OK` only if the solution is ready. Any other code means that the solution is not ready.

#### `POST /take_exam`

It is a request to begin the exam. Body of the request is a JSON object with an instance of exam test in JSON format (a sample JSON could be found in the folder `test_data`).

The solution should response to this request `200 OK` and return a JSON-object with answers to the tasks.  

Both the request and the response should have `Content-Type: application/json`. We recommend to use UTF-8 encoding.

We also publish a file `metadata.json` which was used for submission. Its content is below:

```json
{
    "image": "alenush25/combined_solution_aij:latest",
    "entry_point": "python solution.py"
}
```

Where `image` — a field with docker-image name for the solution image, `entry_point` — a command which runs the solution. 
As a root directory the root of an archive with solution will be used.

A file `eval_docker.py` an example of upload and processing of an exam instance in JSON from the directory `test_data`. It then is sent to the solution

## Solution Description

The task and solution description could be found [here](solver_description.md).
