# TfIdf

TfIdf retriever based on cosine similarities.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **documents** (*List[Dict[str, str]]*) – defaults to `None`

    Documents in TFIdf retriever are static. The retriever must be reseted to index new documents.

- **tfidf** (*sklearn.feature_extraction.text.sparse.TfidfVectorizer*) – defaults to `None`

    sparse.TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.

- **k** (*Optional[int]*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **batch_size** (*int*) – defaults to `1024`

- **fit** (*bool*) – defaults to `True`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from lenlp import sparse

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
...     {"id": 1, "title": "Montreal", "article": "Montreal is in Canada."},
...     {"id": 2, "title": "Paris", "article": "Eiffel tower"},
...     {"id": 3, "title": "Montreal", "article": "Montreal is in Canada."},
... ]

>>> retriever = retrieve.TfIdf(
...     key="id",
...     on=["title", "article"],
...     documents=documents,
... )

>>> documents = [
...     {"id": 4, "title": "Paris", "article": "Eiffel tower"},
...     {"id": 5, "title": "Montreal", "article": "Montreal is in Canada."},
...     {"id": 6, "title": "Paris", "article": "Eiffel tower"},
...     {"id": 7, "title": "Montreal", "article": "Montreal is in Canada."},
... ]

>>> retriever = retriever.add(documents)

>>> print(retriever(q=["paris", "canada"], k=4))
[[{'id': 6, 'similarity': 0.5404109029445249},
  {'id': 0, 'similarity': 0.5404109029445249},
  {'id': 2, 'similarity': 0.5404109029445249},
  {'id': 4, 'similarity': 0.5404109029445249}],
 [{'id': 7, 'similarity': 0.3157669764669935},
  {'id': 5, 'similarity': 0.3157669764669935},
  {'id': 3, 'similarity': 0.3157669764669935},
  {'id': 1, 'similarity': 0.3157669764669935}]]

>>> print(retriever(["unknown", "montreal paris"], k=2))
[[],
 [{'id': 7, 'similarity': 0.7391866872635209},
  {'id': 5, 'similarity': 0.7391866872635209}]]

>>> print(retriever(q="paris"))
[{'id': 6, 'similarity': 0.5404109029445249},
 {'id': 0, 'similarity': 0.5404109029445249},
 {'id': 2, 'similarity': 0.5404109029445249},
 {'id': 4, 'similarity': 0.5404109029445249}]
```

## Methods

???- note "__call__"

    Retrieve documents from batch of queries.

    **Parameters**

    - **q**     (*Union[str, List[str]]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **kwargs**    
    
???- note "add"

    Add new documents to the TFIDF retriever. The tfidf won't be refitted.

    **Parameters**

    - **documents**     (*list*)    
        Documents in TFIdf retriever are static. The retriever must be reseted to index new documents.
    - **batch_size**     (*int*)     – defaults to `100000`    
    - **tqdm_bar**     (*bool*)     – defaults to `False`    
    - **kwargs**    
    
???- note "top_k"

    Return the top k documents for each query.

    **Parameters**

    - **similarities**     (*scipy.sparse._csc.csc_matrix*)    
    - **k**     (*int*)    
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.
    
## References

1. [sklearn.feature_extraction.text.sparse.TfidfVectorizer](https://github.com/raphaelsty/LeNLP)
2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

