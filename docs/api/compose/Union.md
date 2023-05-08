# Union

Union gathers retrieved documents from multiples retrievers and ranked documents from multiples rankers. The union operator concat results with respect of the orders of the models in the union.



## Parameters

- **models** (*list*)

    List of models of the union.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...    {"id": 0, "town": "Paris", "country": "France", "continent": "Europe"},
...    {"id": 1, "town": "Montreal", "country": "Canada", "continent": "North America"},
...    {"id": 2, "town": "Madrid", "country": "Spain", "continent": "Europe"},
... ]

>>> search = (
...     retrieve.TfIdf(key="id", on="town", documents=documents) |
...     retrieve.TfIdf(key="id", on="country", documents=documents) |
...     retrieve.Flash(key="id", on="continent")
... )

>>> search = search.add(documents)

>>> print(search("Paris"))
[{'id': 0, 'similarity': 1.0}]

>>> print(search(["Paris", "Europe"]))
[[{'id': 0, 'similarity': 1.0}],
[{'id': 0, 'similarity': 1.0}, {'id': 2, 'similarity': 0.5}]]
```

## Methods

???- note "__call__"

    

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

