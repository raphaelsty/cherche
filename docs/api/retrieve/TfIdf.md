# TfIdf

TfIdf retriever based on cosine similarities.



## Parameters

- **on** (*str*)

    Field to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **tfidf** (*sklearn.feature_extraction.text.TfidfVectorizer*) – defaults to `None`

    TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> retriever = retrieve.TfIdf(on="title", k=2)

>>> documents = [
...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers.", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch.", "date": "22-11-2021"},
...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers.", "date": "22-11-2020"},
... ]

>>> retriever = retriever.add(documents=documents)

>>> retriever
TfIdf retriever
     on: title
     documents: 3

>>> print(retriever(q="Github"))
[{'date': '22-11-2021',
  'title': 'Github Library with PyTorch.',
  'url': 'mkb/github.com'},
 {'date': '10-11-2021',
  'title': 'Github library with PyTorch and Transformers.',
  'url': 'ckb/github.com'}]
```

## Methods

???- note "__call__"

    Retrieve the right document.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    Add documents to the retriever.

    **Parameters**

    - **documents**     (*list*)    
    
## References

1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

