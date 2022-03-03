# Deployment

Cherche provides a minimalist tool to deploy a neural search pipeline using FastAPI and Docker.

It is a container that hosts an API with two routes:

- A route dedicated to the execution of the neural search pipeline (search).
- A route dedicated to updating the pipeline and the corpus documents (upload).

This tool is compatible with every object of Cherche except Elasticsearch retriever.

## Download and run

The first step consists of downloading the Docker container. The container's name includes the version of Cherche, i.e., `cherche:x.x.x`.

We can check the version of Cherche with the following code.

```python
>>> from cherche import __version__
>>> print(__version__.VERSION) 
(0, 0, 6)
```

Since we are using Cherche (0.0.6), we will download the container raphaelsourty/cherche:0.0.6.

```sh
docker pull raphaelsourty/cherche:0.0.6
```

After downloading the container, we can just run it using the Docker client. We could, for example, add the option `--restart unless-stopped` to restart the container automatically.

```sh
docker run -d --name container -p 80:80 raphaelsourty/cherche:0.0.6
```

## Upload

Once we have downloaded and launched the container, we can upload our model using Cherche. Our version of Cherche should match the one in the container.

```python
>>> from cherche import retrieve, rank, data
>>> from sentence_transformers import SentenceTransformer
>>> import pickle
>>> import requests

>>> documents = data.load_towns()

>>> retriever = retrieve.TfIdf(key="id", on=["article", "title"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...       key = "id",
...       on = ["title", "article"],
...       encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...       k = 10,
... )

>>> search = retriever + ranker + documents
>>> search.add(documents)

>>> requests.post("http://127.0.0.1:80/upload/", files={"model": pickle.dumps(search)})
```

## Search

We can query our container with the model we just uploaded.

```sh
curl "http://127.0.0.1:80/search/?q=Bordeaux"
```

```python
>>> import requests
>>> import urllib
>>> q = "Bordeaux food"
>>> r = requests.get(f"http://127.0.0.1:80/search/?q={urllib.parse.quote(q)}")
>>> r.json()
```

## Customize the API and the Docker container

The source code of the API is available [here](https://github.com/raphaelsty/cherche-api.git).

```sh
git clone https://github.com/raphaelsty/cherche-api.git
cd cherche-api
pip install -r requirements.txt 
```

We can update the API and the Dockerfile and build our container.

```sh
docker build -t cherche:0.0.6 .
docker run -d --name container -p 80:80 cherche:0.0.6
```

## Security concerns

We built the API with the FastAPI framework, and we do not provide an authentication system. To enhance the repository's security, we recommend updating the API code using the recommendations provided [here](https://fastapi.tiangolo.com/tutorial/security/).

The API `upload` route takes a pickle file as input. Therefore, we should not allow an unknown person to upload a pickle file to our API.

## Elasticsearch

If we want to use Elasticsearch, we should update the API code. The Elasticsearch client is not serializable with Pickle and, therefore, must be hard-coded into the API. The connection parameters of the Elasticsearch client should not be hard-coded into the API and stay private. Any contribution would be welcome to facilitate the deployment of a neural search pipeline with Elasticsearch as a retriever.
