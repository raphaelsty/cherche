# Vote

Voting operator. Computes the score for each document based on it's number of occurences and based on documents ranks: $nb_occurences * sum_{rank \in ranks} 1 / rank$. The higher the score, the higher the document is ranked in output of the vote.



## Parameters

- **models** (*list*)

    List of models of the vote.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "town": "Paris", "country": "France", "continent": "Europe"},
...    {"id": 1, "town": "Montreal", "country": "Canada", "continent": "North America"},
...    {"id": 2, "town": "Madrid", "country": "Spain", "continent": "Europe"},
... ]

>>> search = (
...     retrieve.TfIdf(key="id", on="town", documents=documents) *
...     retrieve.TfIdf(key="id", on="country", documents=documents) *
...     retrieve.Flash(key="id", on="continent")
... )

>>> search = search.add(documents)

>>> retriever = retrieve.TfIdf(key="id", on=["town", "country", "continent"], documents=documents)

>>> ranker = rank.Encoder(
...     key="id",
...     on=["town", "country", "continent"],
...     encoder=SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
... ) * rank.Encoder(
...     key="id",
...     on=["town", "country", "continent"],
...     encoder=SentenceTransformer(
...        "sentence-transformers/multi-qa-mpnet-base-cos-v1"
...     ).encode,
... )

>>> search = retriever + ranker

>>> search = search.add(documents)

>>> print(search("What is the capital of Canada ? Is it paris, montreal or madrid ?"))
[{'id': 1, 'similarity': 2.5},
 {'id': 0, 'similarity': 1.4},
 {'id': 2, 'similarity': 1.0}]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*Union[List[List[Dict[str, str]]], List[Dict[str, str]]]*)    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **documents**     (*Optional[List[Dict[str, str]]]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add new documents.

    **Parameters**

    - **documents**     (*list*)    
    - **kwargs**    
    
???- note "reset"

