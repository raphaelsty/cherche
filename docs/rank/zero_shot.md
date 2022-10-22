# rank.ZeroShot

Pre-trained models for the zero-shot classification task are available on
[Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification). There are more than fifty models for different languages. We can find some details [here](https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681).

The `ranker.ZeroShot` model is slow because no pre-computation is possible. Therefore, we strongly recommend using this ranker with a GPU. `ranker.ZeroShot` has no `add` method since it does not pre-compute embeddings.

## Documents

The zero-shot model needs the content of the documents to re-rank the retriever's output candidates. We can provide the documents to the `ranker.ZeroShot` using a pipeline:

```python
>>> search = retriever + documents + ranker
```

Elasticsearch returns the content of the documents by default and not just the identifiers, so there is no need to add the documents to the pipeline with Elasticsearch.

## Quick start

Here, we create a pipeline made of a TfIdf with 30 first documents retrieved, followed by a mapping between identifiers and documents, and finally a zero-shot classifier that keeps five first-order documents.

```python
>>> from cherche import retrieve, rank
>>> from transformers import pipeline

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

>>> ranker = rank.ZeroShot(
...     key = "id",
...     on = "article",
...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
...     k = 5,
... )

# Zero shot needs documents in input.
>>> search = retriever + documents + ranker

>>> search("Paris food")
[{'id': 1,
  'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.8847067356109619},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.5245689749717712},
 {'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.5040559768676758}]
```
