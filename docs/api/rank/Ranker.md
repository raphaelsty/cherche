# Ranker

Abstract class for ranking models.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields of the documents to use for ranking.

- **encoder**

    Encoding function to computes embeddings of the documents.

- **k** (*int*)

    Number of documents to keep.

- **path** (*str*)

    Path of the file dedicated to store the embeddings as a pickle file.

- **similarity**

    Similarity measure to use i.e similarity.cosine or similarity.dot.


## Attributes

- **type**



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

    - **documents**     (*list*)    
    
???- note "dump_embeddings"

    Dump embeddings to the selected directory.

    **Parameters**

    - **embeddings**    
    - **path**     (*str*)    
        Path of the file dedicated to store the embeddings as a pickle file.
    
???- note "embs"

    Computes and returns embeddings of input documents.

    **Parameters**

    - **documents**     (*list*)    
    
???- note "load_embeddings"

    Load embeddings from an existing directory.

    - **path**     (*str*)    
        Path of the file dedicated to store the embeddings as a pickle file.
    
