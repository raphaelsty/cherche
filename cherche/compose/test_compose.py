import pytest

from .. import rank, retrieve


def cherche_retrievers(on: str, k: int = None):
    """List of retrievers available in cherche."""
    yield from [
        retrieve.TfIdf(on=on, k=k),
        retrieve.BM25Okapi(on=on, k=k),
        retrieve.BM25L(on=on, k=k),
    ]


def cherche_rankers(on: str, k: int = None, path: str = None):
    """List of rankers available in cherche."""
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    yield from [
        rank.DPR(
            encoder=SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base").encode,
            query_encoder=SentenceTransformer(
                "facebook-dpr-question_encoder-single-nq-base"
            ).encode,
            on=on,
            k=k,
            path=path,
        ),
        rank.Encoder(
            encoder=SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
            on="title",
            k=k,
            path=path,
        ),
        rank.ZeroShot(
            pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
            on=on,
            k=k,
        ),
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
    "search, documents, k",
    [
        pytest.param(
            retriever + ranker,
            documents(),
            k,
            id=f"retriever: {retriever.__class__.__name__}, ranker: {ranker.__class__.__name__}, k: {k}",
        )
        for k in [None, 0, 2, 4]
        for ranker in cherche_rankers(on="title", k=k, path=None)
        for retriever in cherche_retrievers(on="title", k=k)
    ],
)
def test_retriever_ranker(search, documents: list, k: int):
    """Test retriever ranker pipeline. Test if the number of retrieved documents is coherent.
    Check if the number of documents asked is higher than the actual number of documents retrieved.
    Check if the retriever do not find any document.
    """
    search = search.add(documents)

    answers = search(q="Github library with PyTorch and Transformers.")
    for index in range(len(documents) if k is None else k):
        if index in [0, 1]:
            assert answers[index]["title"] == "Github library with Pytorch and Transformers."
        elif index in [2]:
            assert answers[index]["title"] == "Github Library with PyTorch."

    answers = search(q="Github")
    if k is None:
        assert len(answers) == len(documents)
    else:
        assert len(answers) == min(k, len(documents))

    for sample in answers:
        for key in ["url", "title", "date"]:
            assert key in sample


@pytest.mark.parametrize(
    "search, documents, k",
    [
        pytest.param(
            retrieve.Flash(on="tags", k=k) + ranker,
            tags(),
            k,
            id=f"retriever: Flash, {ranker.__class__.__name__}, k: {k}",
        )
        for k in [None, 0, 2, 4]
        for ranker in cherche_rankers(on="title", k=k, path=None)
    ],
)
def test_flash_ranker(search, documents: list, k: int):
    search = search.add(documents)
    answers = search(q="Github ( git ) is a great tool")
    if k is None:
        assert len(answers) == 2
    else:
        assert len(answers) == min(k, 2)
    for sample in answers:
        for key in ["tags", "title", "uri"]:
            assert key in sample
