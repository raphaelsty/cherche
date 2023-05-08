# Rank

Rankers are models that measure the semantic similarity between a document and a query. Rankers filter out documents based on the semantic similarity between the query and the documents. Rankers are compatible with all the retrievers.


|      Ranker     | Precomputing |                                                          GPU                                                          |
|:---------------:|:------------:|:---------------------------------------------------------------------------------------------------------------------:|
|  ranker.Encoder |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
|    ranker.DPR   |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
| ranker.CrossEncoder |       ❌      |                     Highly recommended since <br>ranker.ZeroShot cannot precompute <br>embeddings                     |
| ranker.Embedding |       ❌      |                    Not needed                     |


The `rank.Encoder` and `rank.DPR` rankers pre-compute the document embeddings once for all with the `add` method. This step can be time-consuming if we do not have a GPU. The embeddings are pre-computed so that the model can then rank the retriever documents at lightning speed.

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
... )

# Pre-compute embeddings
>>> ranker.add(documents, batch_size=64)

>>> match = retriever(["paris", "art", "fashion"], k=100)

# Re-rank output of retriever
>>> ranker(["paris", "art", "fashion"], documents=match, k=30)
[[{'id': 0, 'similarity': 0.6638489},
  {'id': 2, 'similarity': 0.602515},
  {'id': 1, 'similarity': 0.60133684}],
 [{'id': 1, 'similarity': 0.10321068}],
 [{'id': 1, 'similarity': 0.26405674}, {'id': 2, 'similarity': 0.096046045}]]
```
