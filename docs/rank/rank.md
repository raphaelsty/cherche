# Rank

Rankers are models that measure the semantic similarity between a document and a query. The ranker
allows to reorder the documents retrieved by the retriever based on the semantic similarity between
the query and the documents retrieved. Rankers are compatible with all the retrievers.

## key, on and k parameters

The `key` parameter is mandatory for `ranker.Encoder` and `ranker.DPR` but not needed for
`ranker.ZeroShot`. The `key` parameter is the unique identifier of the documents in the corpus.

The `on` parameter allows the ranker to be used on multiple fields. Rankers will concatenate selected
fields to calculate the embeddings of the documents.

Rankers all have a `k`-parameter during the initialization which allows to select the number of
documents to keep after the ranking. When the `k` parameter is set to the default value, i.e. `None`,
the ranker will not drop any documents, it will only reorder them.

|      Ranker     | Precomputing |                                                          GPU                                                          |
|:---------------:|:------------:|:---------------------------------------------------------------------------------------------------------------------:|
|  ranker.Encoder |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
|    ranker.DPR   |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
| ranker.ZeroShot |       ❌      |                     Highly recommended since <br>ranker.ZeroShot cannot precompute <br>embeddings                     |

The `rank.Encoder` and `rank.DPR` rankers pre-compute the document embeddings once for all with the `add` method.
This step can be time consuming if you don't have a GPU. The embeddings are pre-computed so that the model can then rank the retriever documents at lightning speed.
The embeddings can be saved in `pickle` format via the `path` parameter when the ranker is initialized.
At a new initialization the model will use the pre-computed embeddings if the `path` parameter is provided.

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
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker 
>>> search.add(documents)
>>> search(q="france")
[{'id': 0, 'similarity': 0.44967225}, {'id': 2, 'similarity': 0.3609671}]
```

## Save ranker embeddings

The `rank.Encoder` and `rank.DPR` have a `path` parameter at initialization. If you specify this
parameter, the ranker will export the embeddings it has calculated into a `pickle` file.
The pre-computed embeddings are a dictionary with the document id as key and the document embedding
as value. You can reload the pre-computed embeddings in a new session by keeping the pickle file and
specifying the `path` parameter with the address of the pickle file. You will still have to call
the `add` method and index all the documents but this step will be quick because you have already
pre-calculated the embddings.

Otherwise to save a ranker it is also possible to serialize it directly with `pickle`.

```python
>>> import pickle
>>> with open("pipeline.pkl", "wb") as output_file:
...    pickle.dump(search, output_file)
```

You can load your pipeline in another session using pickle again.

```python
>>> import pickle
>>> with open("pipeline.pkl", "rb") as input_file:
...    search = pickle.load(input_file)
```
