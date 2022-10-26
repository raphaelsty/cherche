# Ranker

Abstract class for ranking models.



## Parameters

- **key** (*'str'*)

    Field identifier of each document.

- **on** (*'str | list'*)

    Fields of the documents to use for ranking.

- **encoder**

    Encoding function to computes embeddings of the documents.

- **k** (*'int'*)

    Number of documents to keep.

- **similarity**

    Similarity measure to use i.e similarity.cosine or similarity.dot.

- **store**


## Attributes

- **type**



## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*'str'*)    
    - **documents**     (*'list'*)    
    - **kwargs**    
    
???- note "add"

    Pre-compute embeddings and store them at the selected path.

    **Parameters**

    - **documents**     (*'list'*)    
    - **batch_size**     (*'int'*)     – defaults to `64`    
    
???- note "encode"

    Computes documents embeddings.

    **Parameters**

    - **documents**     (*'list'*)    
    
???- note "rank"

    Rank inputs documents ordered by relevance among the top k.

    **Parameters**

    - **query_embedding**     (*'np.ndarray'*)    
    - **documents**     (*'list'*)    
    
