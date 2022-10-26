# Recommend

The Recommend ranker integrates collaborative filtering models such as matrix factorization in an information retrieval pipeline. It is helpful if you have a history associated with users and can be used to complete the search with the documents most likely to interest the user.

Collaborative filtering models produce embeddings from a matrix of interaction between users and items (documents). Example of a matrix "documents visited by users":

|            | **Document 1** | **Document 2** | **Document 3** | **Document 4** |
|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|
| **User 1** |        1       |                |                |        1       |
| **User 2** |                |        1       |                |                |
| **User 3** |                |        4       |        1       |                |

The Recommend ranker allows to reorder the documents and to select the most relevant documents according to the user's history. The Recommend ranker allows to easily integrate models of [Implicit](https://github.com/benfred/implicit). Not only limited to the library Implicit, but it can also incorporate all the models that represent the user and the documents and whose similarity can be measured via a distance measure of type L2, cosine, or a scalar product.

The Recommend module is also available as a retriever.

## Recommend ranker

```python
>>> from cherche import utils
>>> from implicit.nearest_neighbours import bm25_weight
>>> from implicit.als import AlternatingLeastSquares

>>> documents = [
...    {"id": "document 1", "title": "title 1", "author": "author 1"},
...    {"id": "document 2", "title": "title 2", "author": "author 2"},
...    {"id": "document 3", "title": "title 3", "author": "author 3"},
... ]

>>> visits = {
...    "user 1": {"document 1": 1, "document 4": 1},
...    "user 2": {"document 2": 1},
...    "user 3": {"document 2": 4, "document 3": 1},
... }

>>> model = AlternatingLeastSquares(
...     factors=64, # Dimension of embeddings
...     regularization=0.05,
...     alpha=2.0,
...     iterations=100,
...     random_state=42,
... )

>>> index_users, index_documents, sparse_visits = utils.users_items_sparse(ratings=visits)

>>> model.fit(sparse_visits)

>>> embeddings_users = {
...    user: embedding for user, embedding in zip(index_users, model.user_factors)
... }

>>> embeddings_documents = {
...    document: embedding
...    for document, embedding in zip(index_documents, model.item_factors)
... }
```

`retrieve.Recommend` takes the mapping between user IDs and documents with their embeddings as input where each embedding is a flat numpy array.

```python
>>> from cherche import rank, retrieve

>>> retriever = retrieve.TfIdf(key="id", on=["title", "author"], documents=documents, k=30)

>>> ranker = rank.Recommend(
...    key="id",
...    k=10, # Number of candidates to keeps in the ranking.
... )

>>> pipeline = retriever + ranker + documents

>>> ranker.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
...    embeddings_users=embeddings_users,
... )

>>> pipeline(q="title", user="user 1")
```

```python
[{'id': 'document 1',
  'title': 'title 1',
  'author': 'author 1',
  'similarity': 1.000000238418579},
 {'id': 'document 2',
  'title': 'title 2',
  'author': 'author 2',
  'similarity': 4.411040777085873e-08},
 {'id': 'document 3',
  'title': 'title 3',
  'author': 'author 3',
  'similarity': -1.7862197410067893e-07}]
 ```

## Store

By default, the Recommend ranker store embeddings in memory. The Recommend ranker is compatible with the Milvus database if we do not want to store all the vectors in memory.



