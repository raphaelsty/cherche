# TfIdf

TfIdf retriever based on cosine similarities.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **documents** (*List[Dict[str, str]]*)

    Documents in TFIdf retriever are static. The retriever must be reseted to index new documents.

- **tfidf** (*sklearn.feature_extraction.text.TfidfVectorizer*) – defaults to `None`

    TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.

- **k** (*Optional[int]*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **batch_size** (*int*) – defaults to `1024`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
...     {"id": 1, "title": "Paris", "article": "Paris is in France."},
...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada."},
... ]

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)

>>> retriever
TfIdf retriever
    key      : id
    on       : title, article
    documents: 3

>>> print(retriever(q=["paris", "montreal paris"]))
[[{'id': 1, 'similarity': 0.366173437788525},
  {'id': 0, 'similarity': 0.23008513690129015}],
 [{'id': 2, 'similarity': 0.6568592005036291},
  {'id': 1, 'similarity': 0.18870017418263602},
  {'id': 0, 'similarity': 0.07522017339345569}]]

>>> print(retriever(["unknown", "montreal paris"], k=2))
[[],
 [{'id': 2, 'similarity': 0.6568592005036291},
  {'id': 1, 'similarity': 0.18870017418263602}]]

>>> print(retriever(q="paris", k=2))
[{'id': 1, 'similarity': 0.366173437788525},
 {'id': 0, 'similarity': 0.23008513690129015}]
```

## Methods

???- note "__call__"

    Retrieve documents from batch of queries.

    **Parameters**

    - **q**     (*Union[str, List[str]]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "top_k_by_partition"

    Top k elements by partition.

    **Parameters**

    - **similarities**     (*numpy.ndarray*)    
    - **k**     (*int*)    
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.
    
## References

1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

