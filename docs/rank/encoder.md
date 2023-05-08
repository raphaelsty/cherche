# rank.Encoder

The `rank.Encoder` model re-ranks documents in ouput of the retriever using a pre-trained [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html).

The `rank.Encoder` can pre-compute the set of document embeddings to speed up search and avoiding computing embeddings twice using method `.add`. A GPU will significantly reduce pre-computing time dedicated to document embeddings.

## Requirements

To use the Encoder ranker we will need to install "cherche[cpu]"

```sh
pip install "cherche[cpu]"
```

or on GPU:

```sh
pip install "cherche[gpu]"
```

## Tutorial

```python
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer

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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    normalize=True,
... )

>>> ranker.add(documents, batch_size=64)

>>> match = retriever(["paris", "art", "fashion"], k=100)

# Re-rank output of retriever
>>> ranker(["paris", "art", "fashion"], documents=match, k=30)
[[{'id': 0, 'similarity': 0.6638489}, # Query 1
  {'id': 2, 'similarity': 0.602515},
  {'id': 1, 'similarity': 0.60133684}],
 [{'id': 1, 'similarity': 0.10321068}], # Query 2
 [{'id': 1, 'similarity': 0.26405674}, {'id': 2, 'similarity': 0.096046045}]] # Query 3
```

## Ranker in pipeline

```python
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer

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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=100)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    k = 30,
...    normalize=True,
... )

>>> search = retriever + ranker
>>> search.add(documents)
>>> search(q=["paris", "arts", "fashion"])
[[{'id': 0, 'similarity': 0.6638489}, # Query 1
  {'id': 2, 'similarity': 0.602515},
  {'id': 1, 'similarity': 0.60133684}],
 [{'id': 1, 'similarity': 0.21684976}], # Query 2
 [{'id': 1, 'similarity': 0.26405674}, {'id': 2, 'similarity': 0.096046045}]] # Query 3
```

## Map index to documents

We can map the documents to the ids retrieved by the pipeline.

```python
>>> search += documents
>>> search(q="arts")
[{'id': 1,
  'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.21684971}]
```

## Pre-trained encoders

Here is the list of models provided by [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html). This list of models is not exhaustive; there is a wide range of models available with [Hugging Face](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads) and in many languages.
