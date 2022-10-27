# Retriever

Retriever base class.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*Optional[int]*)

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.


## Attributes

- **type**



## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
