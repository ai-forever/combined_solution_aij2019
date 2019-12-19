# AI Journey 2019: Combined Solution

Комбинация лучших решений топ-20 участников соревнования AI Journey 2019 по каждому заданию. Комбинированное решение набирает 69 баллов по ЕГЭ, русский язык.

## Банк данных и модели

Банк данных AI Journey 2019 содержит данные и материалы, которые могут быть использованы для решения AGI-задач и прикладных NLP-задач:

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
    
Для скачивания банка данных:

```
python download_data.py
```

Каталог ```models``` содержит модели и файлы для решения экзаменационных заданий.

Для скачивания каталога моделей:

```
python download_models.py
```

## Запуск docker контейнера

Запустить контейнер можно командой:
```
$ sudo docker run -w /workspace -v $(pwd):/workspace -p 8000:8000 -it alenush25/combined_solution_aij:latest python solution.py
```

Это поднимет решение, которое представляет собой HTTP-сервер, доступный по порту `8000`, со следующими запросами:

#### `GET /ready`

Запрос отвечает кодом `200 OK` в случае, если решение готово к работе. Любой другой код означает, что решение еще не готово.

#### `POST /take_exam`

Запрос на решение экзаменационного билета. Тело запроса — JSON объект экзаменационного билета в формате JSON соревнования (пример можно найти в папке `test_data`)

Запрос отвечает кодом `200 OK` и возвращает JSON-объект с ответами на задания.  

Запрос и ответ должны иметь `Content-Type: application/json`. Рекомендуется использовать кодировку UTF-8.

Прилагается также аутентичный файл для формата соревнования `metadata.json` следующего содержания:

```json
{
    "image": "alenush25/combined_solution_aij:latest",
    "entry_point": "python solution.py"
}
```

Здесь `image` — поле с названием docker-образа, в котором будет запускаться решение, `entry_point` — команда, при помощи которой запускается решение. Для решения текущей директорией будет являться корень архива.

Файл `eval_docker.py` сожержит пример загрузки и обработки JSON вариантов из папки `test_data` в решение.

## Описание решений

Описание решений и заданий находится [тут](solver_description.md).
