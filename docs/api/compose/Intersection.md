# Intersection

Intersection gathers retrieved documents from multiples retrievers and ranked documents from multiples rankers only if they are proposed by all models of the intersection pipeline.



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
...     retrieve.TfIdf(key="id", on="town", documents=documents) &
...     retrieve.TfIdf(key="id", on="country", documents=documents) &
...     retrieve.Flash(key="id", on="continent")
... )

>>> search = search.add(documents)

>>> print(search("Paris"))
[]

>>> print(search(["Paris", "Europe"]))
[[], []]

>>> print(search(["Paris", "Europe", "Paris Madrid Europe France Spain"]))
[[],
[],
[{'id': 2, 'similarity': 4.25}, {'id': 0, 'similarity': 3.0999999999999996}]]
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

