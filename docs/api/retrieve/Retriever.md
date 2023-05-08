# Retriever

Retriever base class.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*Optional[int]*)

- **batch_size** (*int*)




## Methods

???- note "__call__"

    Retrieve documents from the index.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **k**     (*Optional[int]*)    
    - **batch_size**     (*Optional[int]*)    
    - **kwargs**    
    
