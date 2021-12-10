# Gpu

## Rank.Encoder and Rank.DPR

Similarity-based models from [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) and [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification) can be run on a GPU. Under the hood, `rank.Encoder` and `rank.DPR` pre-computes document embeddings with the `rank.Encoder.add` and `rank.DPR.add` methods. The GPU significantly speeds up the pre-computation phase of embeddings. The embeddings are then saved using [pickle](https://docs.python.org/3/library/pickle.html) format by specifying the `path` parameter.

During the inference phase i.e. search, the rankers check if there is a pre-computed embedding for the query and the documents. In principle, the GPU will only be used to encode the user's query, which is fast. It could be useful to use the GPU to save a precious milliseconds and possibly to quickly encode documents for which we would not have asked the ranker to precompute embeddings.

The ranker `rank.ZeroShot` is a wrapper for Zero shot classification pipeline from [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-classification). These models cannot pre-compute embeddings, so it is essential to use a GPU with this ranker to get satisfactory performance.

## Rank.Encoder

Sentence Transformers as an encoder on GPU:

```python
>>> documents = [{"document": "Lorem Ipsum."}, {"document": "Ipsum Lorem."}]

# Ask the model to load and save embeddings at ./encoder.pkl
>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2", device='cuda').encode,
...    on = "document",
...    k = 30,
...    path = "encoder.pkl"
... )

# Pre compute embeddings using GPU.
>>> ranker.add(documents=documents)
```

## Rank.DPR

Dense Passage Retrieval models on GPU:

```python
>>> documents = [{"document": "Lorem Ipsum."}, {"document": "Ipsum Lorem."}]

# Ask the model to load and save embeddings at ./dpr.pkl
>>> ranker = rank.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base', device="cuda").encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base', devica="cuda").encode,
...    on = "title",
...    k = 30,
...    path = "dpr.pkl"
... )

# Pre compute embeddings using GPU.
>>> ranker.add(documents=documents)
```

## rank.ZeroShot

To use the `zero-shot-classification` models with a GPU, the `device` parameter must be specified. By default the parameter `device` is set to -1 to run on cpu. You needs to set it as a positive integer that match your cuda device id to run it on GPU.

```python
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"document": "Lorem ipsum dolor sit amet"},
...    {"document": " Duis aute irure dolor in reprehenderit"},
... ]

>>> ranker = rank.ZeroShot(
...     encoder = pipeline("zero-shot-classification", 
...         model="typeform/distilbert-base-uncased-mnli", 
...         device=0 # cuda:0
...     ), 
...     on = "document",
...     k = 2,
... )
```
