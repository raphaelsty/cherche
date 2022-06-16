__all__ = ["quantize"]

import torch


def quantize(model, dtype=torch.qint8, layers={torch.nn.Linear}, engine="qnnpack"):
    """Quantize model to speedup inference. May reduce accuracy.

    Parameters
    ----------
    model
        Transformer model to quantize.
    dtype
        Dtype to apply to selected layers.
    layers
        Layers to quantize.
    engine
        The qengine specifies which backend is to be used for execution.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import deploy, retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> encoder = deploy.quantize(SentenceTransformer("sentence-transformers/all-mpnet-base-v2"))

    >>> retriever = retrieve.Encoder(
    ...    encoder = encoder.encode,
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    k = 2,
    ... )

    >>> retriever = retriever.add(documents)

    >>> retriever("paris")
    [{'id': 0, 'similarity': 1.3022390663430141}, {'id': 1, 'similarity': 1.1209681961691136}]

    References
    ----------
    1. [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

    """
    torch.backends.quantized.engine = engine
    return torch.quantization.quantize_dynamic(model, layers, dtype=dtype)
