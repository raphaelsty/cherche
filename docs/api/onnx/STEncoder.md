# STEncoder

Encoder dedicated to run Sentence Transformer models with Onnxruntime.



## Parameters

- **session**

    Onnxruntime inference session.

- **tokenizer**

    Transformer dedicated tokenizer.

- **layers** (*'list'*)

- **max_length** (*'typing.Optional[int]'*) – defaults to `None`




## Methods

???- note "encode"

    Sentence transformer encoding function.

    **Parameters**

    - **sentences**     (*'str | list'*)    
    
