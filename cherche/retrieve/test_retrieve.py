import pytest

from .. import rank, retrieve


def cherche_retrievers(on: str, k: int = None):
    """List of retrievers available in cherche."""
    yield from [
        retrieve.TfIdf(on=on, k=k),
        retrieve.BM25Okapi(on=on, k=k),
        retrieve.BM25L(on=on, k=k),
        retrieve.BM25Plus(on=on, k=k),
    ]


def documents():
    return [
        {
            "url": "ckb/github.com",
            "title": "Github library with Pytorch and Transformers.",
            "date": "10-11-2021",
        },
        {"url": "mkb/github.com", "title": "Github Library with PyTorch.", "date": "22-11-2021"},
        {
            "url": "blp/github.com",
            "title": "Github library with Pytorch and Transformers.",
            "date": "22-11-2020",
        },
    ]


def tags():
    return [
        {"tags": ["Github", "git"], "title": "Github is a great tool.", "uri": "tag:github"},
        {
            "tags": ["cherche", "tool"],
            "title": "Cherche is a tool to retrieve informations.",
            "uri": "tag:cherche",
        },
        {
            "tags": ["python", "programming"],
            "title": "Python is a programming Language",
            "uri": "tag:python",
        },
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
        for retriever in cherche_retrievers(on="title", k=k)
    ],
)
def test_retriever(retriever, documents: list, k: int):
    """Test retriever. Test if the number of retrieved documents is coherent.
    Check if the retriever do not find any document.
    """
    retriever = retriever.add(documents)

    # All documents contains Github
    answers = retriever(q="Github")
    if k is None:
        assert len(answers) == len(documents)
    elif k == 0:
        assert len(answers) == 0
    else:
        assert len(answers) == min(k, len(documents))

    for sample in answers:
        for key in ["url", "title", "date"]:
            assert key in sample

    # Unknown token.
    answers = retriever(q="Unknown")
    assert len(answers) == 0
