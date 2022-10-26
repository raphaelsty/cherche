# Recommend

Collaborative filtering as a retriever. Recommend is compatible with the library [Implicit](https://github.com/benfred/implicit).



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **index** – defaults to `None`

    Index that will store the embeddings of documents and perform the similarity search. The default index is Faiss. We can choose index.Milvus also.

- **store** – defaults to `<cherche.retrieve.recommend.MemoryStore object at 0x168b6ca30>`

    Index that will store the embeddings of users. By default, it store users embeddings in memory. We can choose index.Milvus also.


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve, utils
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

>>> recommend = retrieve.Recommend(
...    key="id",
...    k = 10,
... )

>>> recommend.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
...    embeddings_users=embeddings_users,
... )
Recommend retriever
    key: id
    Users: 4
    Documents: 3

>>> recommend += documents

>>> print(recommend(user="Geoffrey"))
[{'author': 'Montreal',
  'id': 'c',
  'similarity': 21229.834241369794,
  'title': 'Montreal'},
 {'author': 'Paris',
  'id': 'a',
  'similarity': 21229.634204933023,
  'title': 'Paris'},
 {'author': 'Madrid',
  'id': 'b',
  'similarity': 0.5075642957423536,
  'title': 'Madrid'}]
```

## Methods

???- note "__call__"

    Retrieve documents from user id.

    **Parameters**

    - **user**     (*Union[str, int]*)    
    - **expr**     (*str*)     – defaults to `None`    
    - **consistency_level**     (*str*)     – defaults to `None`    
    - **partition_names**     (*list*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add embeddings both documents and users.

    **Parameters**

    - **documents**     (*list*)    
    - **embeddings_documents**     (*dict*)    
    - **embeddings_users**     (*dict*)    
    - **kwargs**    
    
## References

1. [Implicit](https://github.com/benfred/implicit)
2. [Implicit documentation](https://benfred.github.io/implicit/)
3. [Logistic Matrix Factorization for Implicit Feedback Data](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)
4. [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf)
5. [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)

