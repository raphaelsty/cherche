# Pipeline

Neurals search pipeline.



## Parameters

- **models** (*list*)

    List of models of the pipeline.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve, rank, qa, summary
>>> from sentence_transformers import SentenceTransformer
>>> from transformers import pipeline

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retrieve.TfIdf(key="id", on="article", documents=documents)

```

Retriever, Ranker:
```python
>>> ranker = rank.Encoder(
...    on = "article",
...    key = "id",
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    path = "pipeline_encoder.pkl"
... )

>>> search = retriever + ranker + documents

>>> search.add(documents=documents)
TfIdf retriever
     key: id
     on: article
     documents: 3
Encoder ranker
     key: id
     on: article
     k: None
     similarity: cosine
     embeddings stored at: pipeline_encoder.pkl
Mapping to documents

>>> print(search(q = "Paris"))
[{'article': 'Paris is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.7014109,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.51787204,
  'title': 'Eiffel tower'}]

```

Retriever, Ranker, Question Answering:
```python
>>> search += qa.QA(
...     model = pipeline(
...         "question-answering",
...         model = "deepset/roberta-base-squad2",
...         tokenizer = "deepset/roberta-base-squad2"
...     ),
...     on = "article",
... )

>>> search
TfIdf retriever
     key: id
     on: article
     documents: 3
Encoder ranker
     key: id
     on: article
     k: None
     similarity: cosine
     embeddings stored at: pipeline_encoder.pkl
Mapping to documents
Question Answering
     model: deepset/roberta-base-squad2
     on: article

>>> print(search(q = "What is based in Paris?"))
[{'answer': 'Eiffel tower',
  'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'end': 12,
  'id': 1,
  'qa_score': 0.9643093347549438,
  'similarity': 0.65787125,
  'start': 0,
  'title': 'Eiffel tower'},
 {'answer': 'Paris is the capital of France',
  'article': 'Paris is the capital of France',
  'author': 'Wiki',
  'end': 30,
  'id': 0,
  'qa_score': 4.247476681484841e-05,
  'similarity': 0.7062913,
  'start': 0,
  'title': 'Paris'},
 {'answer': 'Montreal is in Canada.',
  'article': 'Montreal is in Canada.',
  'author': 'Wiki',
  'end': 22,
  'id': 2,
  'qa_score': 1.7172554933608808e-08,
  'similarity': 0.3316515,
  'start': 0,
  'title': 'Montreal'}]
```

## Methods

???- note "__call__"

    Compose pipeline

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

???- note "reset"

