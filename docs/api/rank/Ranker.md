# Ranker

Abstract class for ranking models.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, List[str]]*)

    Fields of the documents to use for ranking.

- **encoder**

    Encoding function to computes embeddings of the documents.

- **normalize** (*bool*)

    Normalize the embeddings in order to measure cosine similarity if set to True, dot product if set to False.

- **batch_size** (*int*)

- **k** (*Optional[int]*) – defaults to `None`




## Methods

???- note "__call__"

    Rank documents according to the query.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **documents**     (*Union[List[List[Dict[str, str]]], List[Dict[str, str]]]*)    
    - **k**     (*int*)    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Pre-compute embeddings and store them at the selected path.

    **Parameters**

    - **documents**     (*List[Dict[str, str]]*)    
    - **batch_size**     (*int*)     – defaults to `64`    
    
???- note "encode_rank"

    Encode documents and rank them according to the query.

    **Parameters**

    - **embeddings_queries**     (*numpy.ndarray*)    
    - **documents**     (*List[List[Dict[str, str]]]*)    
    - **k**     (*int*)    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    
???- note "rank"

    Rank inputs documents ordered by relevance among the top k.

    **Parameters**

    - **embeddings_documents**     (*Dict[str, numpy.ndarray]*)    
    - **embeddings_queries**     (*numpy.ndarray*)    
    - **documents**     (*List[List[Dict[str, str]]]*)    
    - **k**     (*int*)    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    
