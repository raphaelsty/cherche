# rank.DPR

`rank.DPR` is dedicated to the [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) models
which aims to train two distinct neural networks, one that encodes the query and the other one that
encodes the documents.

## Pre-compute

The `rank.DPR` can pre-compute the set of document embeddings to speed up search in the
production environment. To store the embeddings of the ranker, it is necessary to to specify the
`path` parameter. The document embeddings will then be stored in pickle format and at the specified address
as a dictionary with the document identifier as key and the document embedding as value. `rank.Encoder`
will load the embeddings from the `path` file when calling the `add` function. If the embedding is not already
already calculated for a document, it will calculate it once for all. A GPU will significantly
improves the time to pre-compute embeddings otherwise this step is slow.

## Quick Start

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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> ranker = rank.DPR(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    k = 5,
...    path = "dpr.pkl"
... )

>>> search = retriever + ranker
>>> search.add(documents)
>>> search("France")
[{'id': 0, 'similarity': 72.57037}, {'id': 2, 'similarity': 70.165115}]
```

## Pre-trained DPR

From a personal point of view, I have found that results of the DPR model are regularly less
relevant than Sentence Transformer results.

|               Question Encoder               |             Document encoder            |
|:--------------------------------------------:|:---------------------------------------:|
|  facebook-dpr-question_encoder-multiset-base |  facebook-dpr-ctx_encoder-multiset-base |
| facebook-dpr-question_encoder-single-nq-base | facebook-dpr-ctx_encoder-single-nq-base |

## Map index to documents

Optionally, you can map the documents to the ids retrieved by the pipeline.

```python
>>> search += documents
>>> search(q="France")
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 72.57037},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 70.165115}]
```

## Custom DPR

You can train your own DPR using the tool of your choice and use it with Cherche.
Your own DPR should have an API similar to the DPR model of Sentence Transformers models. It should
be a function which encodes a list of strings to return a numpy array with dimensions
`(number of documents, embedding size)`. This same function must be able to encode a single string
to return an embedding of size `(1, embedding dimension)`.
