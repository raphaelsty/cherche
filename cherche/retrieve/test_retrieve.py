import pytest

from .. import rank, retrieve


def cherche_retrievers(on: str, k: int = None):
    """List of retrievers available in cherche."""
    yield from [
        retrieve.TfIdf(key="title", on=on, documents=documents(), k=k),
        retrieve.BM25Okapi(key="title", on=on, documents=documents(), k=k),
        retrieve.BM25L(key="title", on=on, documents=documents(), k=k),
        retrieve.Lunr(key="title", on=on, documents=documents(), k=k),
    ]


def documents():
    return [
        {"title": "Paris", "article": "This town is the capital of France", "author": "Wikipedia"},
        {
            "title": "Eiffel tower",
            "article": "Eiffel tower is based in Paris",
            "author": "Wikipedia",
        },
        {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wikipedia"},
    ]


@pytest.mark.parametrize(
    "retriever, documents, k",
    [
        pytest.param(
            retriever,
            documents(),
            k,
            id=f"retriever: {retriever.__class__.__name__}, k: {k}",
        )
        for k in [None, 0, 2, 4]
        for retriever in cherche_retrievers(on="article", k=k)
    ],
)
def test_retriever(retriever, documents: list, k: int):
    """Test retriever. Test if the number of retrieved documents is coherent.
    Check for unknown tokens in the corpus, should returns an empty list.
    """
    retriever = retriever + documents
    retriever.add(documents)

    # A single document contains town.
    answers = retriever(q="town")
    if k is None or k >= 1:
        assert len(answers) == 1
    else:
        assert len(answers) == 0

    for sample in answers:
        for key in ["title", "article", "author"]:
            assert key in sample

    # Unknown token.
    answers = retriever(q="Unknown")
    assert len(answers) == 0

    # All documents contains "Montreal Eiffel France"
    answers = retriever(q="Montreal Eiffel France")
    if k is None or k >= len(documents):
        assert len(answers) == len(documents)
    else:
        assert len(answers) == k


@pytest.mark.parametrize(
    "retriever, documents, k",
    [
        pytest.param(
            retriever,
            documents(),
            k,
            id=f"Multiple fields retriever: {retriever.__class__.__name__}, k: {k}",
        )
        for k in [None, 0, 2, 4]
        for retriever in cherche_retrievers(on=["article", "title", "author"], k=k)
    ],
)
def test_fields_retriever(retriever, documents: list, k: int):
    """Test retriever when providing multiples fields."""
    retriever = retriever + documents

    # All documents have Wikipedia as author.
    answers = retriever(q="Wikipedia")
    if k is None or k >= len(documents):
        assert len(answers) == len(documents)
    else:
        assert len(answers) == max(k, 0)

    for sample in answers:
        for key in ["title", "article", "author"]:
            assert key in sample

    # Unknown token.
    answers = retriever(q="Unknown")
    assert len(answers) == 0

    # Two documents contains paris
    answers = retriever(q="Paris")

    if k is None or k >= 2:
        assert len(answers) == 2
    else:
        assert len(answers) == max(k, 0)


@pytest.mark.parametrize(
    "documents, k",
    [
        pytest.param(
            documents(),
            k,
            id=f"retriever: Flash, k: {k}",
        )
        for k in [None, 0, 2, 4]
    ],
)
def test_flash(documents: list, k: int):
    """Test Flash retriever."""
    # Reset retriever
    retriever = retrieve.Flash(key="title", k=k, on="title") + documents
    retriever.add(documents)

    # A single document contains town.
    answers = retriever(q="Paris")
    if k is None or k >= 1:
        assert len(answers) == 1
    else:
        assert len(answers) == 0

    for sample in answers:
        for key in ["title", "article", "author"]:
            assert key in sample

    # Unknown token.
    answers = retriever(q="Unknown")
    assert len(answers) == 0

    # All documents contains is
    answers = retriever(q="Paris Eiffel tower Montreal")

    if k is None or k >= len(documents):
        assert len(answers) == len(documents)
    else:
        assert len(answers) == k


@pytest.mark.parametrize(
    "documents, k",
    [
        pytest.param(
            documents(),
            k,
            id=f"retriever: ElasticSearch, k: {k}",
        )
        for k in [None, 0, 2, 4]
    ],
)
def test_elastic(documents, k):
    """Test Elasticsearch if elastic server is running. Test elasticsearch as a retriever for a
    single field and multiple fields. Test if storing"""
    from elasticsearch import Elasticsearch
    from sentence_transformers import SentenceTransformer

    es = Elasticsearch()

    if es.ping():

        ranker = rank.Encoder(
            key="title",
            encoder=SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2",
            ).encode,
            on=["title", "article", "author"],
            k=k,
        )

        retriever = retrieve.Elastic(key="title", on="article", k=k, es=es, index="test_cherche")
        retriever.reset()
        retriever.add(documents)
        test_retriever(retriever=retriever, documents=documents, k=k)

        retriever = retrieve.Elastic(
            key="title", on=["title", "article", "author"], k=k, es=es, index="test_cherche"
        )
        retriever.reset()
        retriever.add(documents)
        test_fields_retriever(retriever, documents=documents, k=k)

        # Store embeddings using Elasticsearch
        retriever.reset()
        retriever.add_embeddings(documents=documents, ranker=ranker)
        pipeline = retriever + ranker
        answers = pipeline("Paris")
        if k is None or k >= 2:
            assert len(answers) == 2
        else:
            assert len(answers) == max(k, 0)
