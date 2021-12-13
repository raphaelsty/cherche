# Retriever

Retriever base class.



## Parameters

- **on** (*str*)

    Field to use to match the query to the documents.

- **k** (*int*)

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.




## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

    Add documents to the retriever.

    - **documents**     (*list*)    
    
