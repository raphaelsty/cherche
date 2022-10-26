# users_items_sparse

Convert dict to scipy csr matrix for Implicit.



## Parameters

- **ratings** (*dict*)

    Dict with users as key and documents with their evaluations as values.



## Examples

```python
>>> from cherche import utils

>>> ratings = {
...    "Max": {"a": 1, "c": 1},
...    "Adil": {"b": 1, "d": 2},
...    "Robin": {"b": 1, "d": 1},
...    "Geoffrey": {"a": 1, "c": 1},
... }

>>> users, documents, matrix = utils.users_items_sparse(ratings=ratings)

>>> users
['Max', 'Adil', 'Robin', 'Geoffrey']

>>> documents
['a', 'c', 'b', 'd']

>>> matrix
<4x4 sparse matrix of type '<class 'numpy.int64'>'
    with 8 stored elements in Compressed Sparse Row format>
```

