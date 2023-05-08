__all__ = ["quantize"]


def quantize(model, dtype=None, layers=None, engine="qnnpack"):
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
    >>> from cherche import utils, retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris France"},
    ...    {"id": 1, "title": "Madrid Spain"},
    ...    {"id": 2, "title": "Montreal Canada"}
    ... ]

    >>> encoder = utils.quantize(SentenceTransformer("sentence-transformers/all-mpnet-base-v2"))

    >>> retriever = retrieve.Encoder(
    ...    encoder = encoder.encode,
    ...    key = "id",
    ...    on = ["title"],
    ... )

    >>> retriever = retriever.add(documents)

    >>> print(retriever("paris"))
    [{'id': 0, 'similarity': 0.6361529519968355},
     {'id': 2, 'similarity': 0.42750324298964354},
     {'id': 1, 'similarity': 0.42645383885361576}]

    References
    ----------
    1. [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "Run pip install cherche[cpu] or pip install cherche[gpu] to use quantize."
        )

    if dtype is None:
        dtype = torch.qint8

    if layers is None:
        layers = {torch.nn.Linear}

    torch.backends.quantized.engine = engine
    return torch.quantization.quantize_dynamic(model, layers, dtype)
