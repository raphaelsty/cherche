# Norvig

Spelling corrector written by Peter Norvig: [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html)



## Parameters

- **on** (*Union[str, List]*)

    Fields to use for fitting the spelling corrector on.

- **lower** (*bool*) – defaults to `True`

- **big** (*bool*) – defaults to `False`

    Use the big.txt provided by the Norvig spelling corrector. Contains english books from the Gutenberg project.


## Attributes

- **type**


## Examples

```python
>>> from cherche import query, data

>>> documents = data.load_towns()

>>> corrector = query.Norvig(on = ["title", "article"], lower=True)

>>> corrector.add(documents)
Query Norvig
     Vocabulary: 967

>>> corrector(q="tha citi af Parisa is in Fronce")
'the city of paris is in france'

>>> corrector = query.Norvig(big=True, on=["title", "article"], lower=False)

>>> corrector.add(documents)
Query Norvig
     Vocabulary: 32790

>>> corrector(q="tha citi af Parisa is in Fronce")
'the city of Paris is in France'
```

## Methods

???- note "__call__"

    Correct spelling errors in a given query.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

    Fit Nervig spelling corrector.

    **Parameters**

    - **documents**     (*Union[List[Dict], str]*)    
    
???- note "correct"

    Most probable spelling correction for word.

    **Parameters**

    - **word**     (*str*)    
    
???- note "reset"

    Wipe dictionary.

    
## References

1. [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html)

