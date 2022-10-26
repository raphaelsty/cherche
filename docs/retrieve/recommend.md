# Recommend

The Recommend retriever integrates collaborative filtering models such as matrix factorization in an information retrieval pipeline. It is helpful if you have a history associated with users and can be used to complete the search with the documents most likely to interest the user.

Collaborative filtering models produce embeddings from a matrix of interaction between users and items (documents). Example of a matrix "documents visited by users":

|            | **Document 1** | **Document 2** | **Document 3** | **Document 4** |
|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|
| **User 1** |        1       |                |                |        1       |
| **User 2** |                |        1       |                |                |
| **User 3** |                |        4       |        1       |                |



The Recommend model allows the recommendation of the `k` documents most likely to appeal to a given user. The Recommend ranker allows to easily integrate models of [Implicit](https://github.com/benfred/implicit). Not only limited to the library Implicit, but it can also incorporate all the models that represent the user and the documents and whose similarity can be measured via a distance measure of type L2, cosine, or a scalar product.

The Recommend module is also available as a ranker.

## Recommend retriever

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
>>> from cherche import retrieve

>>> recommend = retrieve.Recommend(
...    key="id",
...    k=10, # Number of candidates to retrieve.
... )

>>> recommend.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
...    embeddings_users=embeddings_users,
... )

>>> recommend(user="user 1")
```

```python
[{'id': 'document 1', 'similarity': 8.357158966165448},
 {'id': 'document 3', 'similarity': 0.4439385609854764},
 {'id': 'document 2', 'similarity': 0.3698757425749123}]
 ```

### Map to documents

```python
>>> recommend += documents

>>> recommend(user="user 1")
```

```python
[{'id': 'document 1',
  'title': 'title 1',
  'author': 'author 1',
  'similarity': 8.357158966165448},
 {'id': 'document 3',
  'title': 'title 3',
  'author': 'author 3',
  'similarity': 0.4439385609854764},
 {'id': 'document 2',
  'title': 'title 2',
  'author': 'author 2',
  'similarity': 0.3698757425749123}]
```

## Index

By default, the Recommend retriever uses a Faiss index to find the documents most similar to the user's representation. It is possible to select the type of the Faiss index and run it on GPU.

### Faiss-GPU

Here is how to create a Recommend retriever that runs on GPU:

```sh
pip install faiss-gpu
```

```python
>>> from cherche import retrieve
>>> import faiss

>>> d = 64 # Embeddings size.
>>> index = faiss.IndexFlatL2(d)
>>> index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index) # 0 is the id of the GPU

>>> recommend = retrieve.Recommend(
...    key="id",
...    k=10, # Number of candidates to retrieve.
...    index=index,
... )
```

The Recommend retriever is compatible with the Milvus database if we do not want to store all the vectors in memory.

