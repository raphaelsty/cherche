# Summary

The `summary.Summary` module integrates a summarization model at the end of the neural search pipeline. It is a handy tool to summarize a subset of documents that match a query. However, summarization models are slow using CPU and require a GPU to get decent response times.

## On

The `on` parameter allows the selection of the field(s) used to generate a summary.

## Documents

The pipeline must provide the documents and not only the identifiers to the summarization model (except for Elasticsearch, which retrieve documents by default).

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
