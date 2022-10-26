# Recommend

Collaborative filtering as a ranker. Recommend is compatible with the library [Implicit](https://github.com/benfred/implicit).



## Parameters

- **key** (*'str'*)

    Field identifier of each document.

- **k** (*'int'*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **similarity** – defaults to `<function cosine at 0x15efedf70>`

- **store_items** – defaults to `<cherche.rank.base.MemoryStore object at 0x15efe1d30>`

- **store_users** – defaults to `<cherche.rank.base.MemoryStore object at 0x15efe1d90>`


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank, utils
>>> from implicit.nearest_neighbours import bm25_weight
>>> from implicit.als import AlternatingLeastSquares

>>> documents = [
...    {"id": "a", "title": "Paris", "author": "Paris"},
...    {"id": "b", "title": "Madrid", "author": "Madrid"},
...    {"id": "c", "title": "Montreal", "author": "Montreal"},
... ]

>>> ratings = {
...    "Max": {"a": 1, "c": 1},
...    "Adil": {"b": 1, "d": 2},
...    "Robin": {"b": 1, "d": 1},
...    "Geoffrey": {"a": 1, "c": 1},
... }

>>> index_users, index_documents, sparse_ratings = utils.users_items_sparse(ratings=ratings)

>>> model = AlternatingLeastSquares(
...     factors=64,
...     regularization=0.05,
...     alpha=2.0,
...     iterations=100,
...     random_state=42,
... )

>>> model.fit(sparse_ratings)

>>> embeddings_users = {
...    user: embedding for user, embedding in zip(index_users, model.user_factors)
... }

>>> embeddings_documents = {
...    document: embedding
...    for document, embedding in zip(index_documents, model.item_factors)
... }

>>> recommend = rank.Recommend(
...    key="id",
...    k = 10,
... )

>>> recommend.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
...    embeddings_users=embeddings_users,
... )
Recommend ranker
    key: id
    Users: 4
    Documents: 3

>>> print(recommend(user="Geoffrey", documents=documents))
[{'author': 'Paris',
  'id': 'a',
  'similarity': 1.0000001192092896,
  'title': 'Paris'},
 {'author': 'Montreal',
  'id': 'c',
  'similarity': 0.9999998807907104,
  'title': 'Montreal'},
 {'author': 'Madrid',
  'id': 'b',
  'similarity': 4.273452987035853e-07,
  'title': 'Madrid'}]

>>> recommend(user="Geoffrey", documents=[{"id": "unknown", "title": "unknown", "author": "unknown"}])
[{'id': 'unknown', 'title': 'unknown', 'author': 'unknown', 'similarity': 0}]

>>> print(recommend(user="unknown", documents=documents))
[{'author': 'Paris', 'id': 'a', 'similarity': 0, 'title': 'Paris'},
 {'author': 'Madrid', 'id': 'b', 'similarity': 0, 'title': 'Madrid'},
 {'author': 'Montreal', 'id': 'c', 'similarity': 0, 'title': 'Montreal'}]
```

## Methods

???- note "__call__"

    Retrieve documents from user id.

    **Parameters**

    - **user**     (*'typing.Union[str, int]'*)    
    - **documents**     (*'list'*)    
    - **expr**     (*'str'*)     – defaults to `None`    
    - **consistency_level**     (*'str'*)     – defaults to `None`    
    - **partition_names**     (*'list'*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add embeddings both documents and users.

    **Parameters**

    - **documents**     (*'list'*)    
    - **embeddings_documents**     (*'dict'*)    
    - **embeddings_users**     (*'dict'*)    
    - **kwargs**    
    
???- note "encode"

    Computes documents embeddings.

    **Parameters**

    - **documents**     (*'list'*)    
    
???- note "rank"

    Rank inputs documents ordered by relevance among the top k.

    **Parameters**

    - **query_embedding**     (*'np.ndarray'*)    
    - **documents**     (*'list'*)    
    
## References

1. [Implicit](https://github.com/benfred/implicit)
2. [Implicit documentation](https://benfred.github.io/implicit/)
3. [Logistic Matrix Factorization for Implicit Feedback Data](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)
4. [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf)
5. [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)

