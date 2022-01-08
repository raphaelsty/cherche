# Summary

The `summary.Summary` module allows to integrate a summarization model at the end of the neural
search pipeline. This is a very handy tool to get a summary of a subset of documents that match a query.

Summarization models are slow using CPU and require a GPU to get decent response times.

## On

The `on` parameter allows to select the field(s) used to generate a summary.

## Documents

It is mandatory that the pipeline provide the documents and not only the identifiers to the
summarization model such as (except for Elasticsearch which retrieve documents by default):

```python
search = pipeline + documents + summarization
```

## Quick Start

```python
>>> from cherche import data, rank, retrieve, qa
>>> from sentence_transformers import SentenceTransformer
>>> from transformers import pipeline

>>> documents = data.load_towns()

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k = 30)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = "article",
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 3,
...    path = "encoder.pkl"
... )

>>> question_answering = qa.QA(
...    model = pipeline("question-answering", 
...         model = "deepset/roberta-base-squad2", 
...         tokenizer = "deepset/roberta-base-squad2"
...    ),
...    on = "article",
... )

>>> search = retriever + ranker + documents + question_answering
>>> search.add(documents)
# Paris Saint-Germain is the answer.
>>> search("What is the name of the football club of Paris?")
[{'start': 18,
  'end': 37,
  'answer': 'Paris Saint-Germain',
  'qa_score': 0.9848363399505615,
  'id': 20,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The football club Paris Saint-Germain and the rugby union club Stade Français are based in Paris.',
  'similarity': 0.7104821},
 {'start': 15,
  'end': 17,
  'answer': '12',
  'qa_score': 0.015906214714050293,
  'id': 16,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'Paris received 12.',
  'similarity': 0.46774143},
 {'start': 29,
  'end': 35,
  'answer': '\u200b[paʁi',
  'qa_score': 2.7218469767831266e-05,
  'id': 0,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'Paris (French pronunciation: \u200b[paʁi] (listen)) is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles).',
  'similarity': 0.52439684}]
```
