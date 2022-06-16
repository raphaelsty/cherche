# ONNX Sentence Transformer

[Open Neural Network Exchange](https://github.com/onnx/onnx) (ONNX) is an open-source solution for accelerating model inference. Cherche brings compatibility between the ONNX format to accelerate Sentence Transformers (retriever and ranker) and question-answering models:

- retriever.Encoder
- ranker.Encoder
- qa.QA

The acceleration includes the transformation to the ONNX format and the quantization of the model. Quantizing a model involves replacing part of the model weights with integers (vs. floats initially). Quantization may reduce the model's accuracy but will significantly accelerate the model.

## Requirements

### ONNX CPU

We can benefit from executing models via the ONNX environment on the CPU; via installing the Cherche library with the `onnx` option:

```sh
!pip install cherche[onnx]
```

### ONNX GPU

We can benefit from executing models via the ONNX environment on the GPU; via installing the Cherche library with the `onnxgpu` option:

```sh
!pip install cherche[onnxgpu]
```

### Mac M1

For MAC M1 owners, you may have difficulties installing ONNX. Here is how to set up compatibility with ONNX:

```sh
conda install onnx
brew install onnxruntime
```

## Sentence Transformers

We can speed up a Sentence Transformer model with the `onnx.sentence_transformers` function. In addition, it is possible to disable quantization via the `quantize=False` parameter.

```python
>>> from cherche import onnx
>>> from sentence_transformers import SentenceTransformer

>>> model = onnx.sentence_transformers(
...    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
...    name = "all-mpnet-base-v2",
...    quantize = True,
...    providers = ["CPUExecutionProvider"], # CPU based runtime
... )
```

We can accelerate the model on GPU:

```python
>>> from cherche import onnx
>>> from sentence_transformers import SentenceTransformer

>>> model = onnx.sentence_transformers(
...    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda"),
...    name = "all-mpnet-base-v2",
...    quantize = True,
...    providers = ["CUDAExecutionProvider"], # GPU based runtime
... )
```

We can then declare our retriever or ranker from the model accelerated by ONNX.

```python
>>> from cherche import retrieve, rank

>>> documents = [
...    {
...        "id": 0,
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 1,
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 2,
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> retriever = retrieve.Encoder(
...    encoder = model.encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 10,
... )

>>> retriever.add(documents)

>>> ranker = rank.Encoder(
...    encoder = model.encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 10,
... )

>>> ranker.add(documents)
```


## Question Answering

```python
>>> from cherche import onnx
>>> from transformers import pipeline

>>> model = onnx.qa(
...    model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2"),
...    name = "roberta-base-squad2",
...    quantize = True,
...    providers = ["CPUExecutionProvider"], # CPU based runtime
... )
```

We can accelerate the model on GPU:

```python
>>> from cherche import onnx
>>> from transformers import pipeline

>>> model = onnx.qa(
...    model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", device=0),
...    name = "roberta-base-squad2",
...    quantize = True,
...    providers = ["CUDAExecutionProvider"], # GPU based runtime
... )
```

We can then integrate the ONNX model into a question answering pipeline:

```python
>>> from cherche import retrieve, qa

>>> documents = [
...    {
...        "id": 0,
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 1,
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 2,
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> question_answering = qa.QA(
...    model = model,
...    on = "article",
...    k = 2,
... )

>>> search = retriever + documents + question_answering

>>> search("capital of france")
[{'start': 4,
  'end': 5,
  'answer': 'Paris',
  'qa_score': 0.48900822,
  'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.5101032403433207},
 {'start': 21,
  'end': 28,
  'answer': 'Île-de-France',
  'qa_score': 0.18752262,
  'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.3009824140535134}]
```

## See also

The library [transformer-deploy](https://github.com/ELS-RD/transformer-deploy) provides tools to speed up the inference of transformers significantly.
