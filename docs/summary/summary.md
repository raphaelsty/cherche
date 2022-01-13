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
>>> from cherche import data, rank, retrieve, summary
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

>>> summarization = summary.Summary(
...    model = pipeline(
...         "summarization",
...         model="sshleifer/distilbart-cnn-6-6",
...         tokenizer="sshleifer/distilbart-cnn-6-6",
...         framework="pt"
...    ),
...    on = ["title", "article"],
... )

>>> search = retriever + ranker + documents + summarization
>>> search.add(documents)

>>> search("Bordeaux wine")
"Bordeaux has been voted European Destination of the year in a 2015 online poll. The region is home to the world's main wine"
```
