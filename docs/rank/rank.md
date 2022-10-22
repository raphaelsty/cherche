# Rank

Rankers are models that measure the semantic similarity between a document and a query. Rankers filter out documents based on the semantic similarity between the query and the documents. Rankers are compatible with all the retrievers.

## key, on and k parameters

The `key` parameter is mandatory for `ranker.Encoder` and `ranker.DPR` but not needed for `ranker.ZeroShot`. The `key` parameter is the unique identifier of the documents in the corpus. We can use ranker on multiple fields with the `on` parameter. Rankers will concatenate selected fields to calculate the embeddings of the documents. The k parameter of ranker allows selecting the number of documents to keep after the ranking. By default, rankers will reorder documents without dropping any.

|      Ranker     | Precomputing |                                                          GPU                                                          |
|:---------------:|:------------:|:---------------------------------------------------------------------------------------------------------------------:|
|  ranker.Encoder |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
|    ranker.DPR   |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
| ranker.ZeroShot |       ❌      |                     Highly recommended since <br>ranker.ZeroShot cannot precompute <br>embeddings                     |

The `rank.Encoder` and `rank.DPR` rankers pre-compute the document embeddings once for all with the `add` method. This step can be time-consuming if we do not have a GPU. The embeddings are pre-computed so that the model can then rank the retriever documents at lightning speed.

## Quick start

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

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
... )

>>> search = retriever + ranker
>>> search.add(documents)
>>> search(q="france")
[{'id': 0, 'similarity': 0.44967225}, {'id': 2, 'similarity': 0.3609671}]
```
