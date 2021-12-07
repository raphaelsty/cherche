# Ranker

Abstract class for ranking models.



## Parameters

- **on** (*str*)

- **encoder**

- **k** (*int*)

- **path** (*str*)

- **metric**




## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **kwargs**    
    
???- note "add"

    Pre-compute embeddings and store them at the selected path.

    **Parameters**

    - **documents**    
    
???- note "dump_embeddings"

    Dump embeddings to the selected directory.

    **Parameters**

    - **embeddings**    
    - **path**     (*str*)    
    
???- note "load_embeddings"

    Load embeddings from an existing directory.

    - **path**     (*str*)    
    
