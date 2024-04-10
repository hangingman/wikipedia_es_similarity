# Text Similarity Search by using Elasticsearch

- [Text similarity search in Elasticsearch using vector fields \| Elastic Blog](https://www.elastic.co/jp/blog/text-similarity-search-with-vectors-in-elasticsearch)
- [jtibshirani/text\-embeddings](https://github.com/jtibshirani/text-embeddings)

## Preparation

- Download docs and embedding

```
$ wget https://dumps.wikimedia.org/other/cirrussearch/20240408/jawiki-20240408-cirrussearch-content.json.gz
$ wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.word_vectors.200d.txt.bz2
$ bunzip2 jawiki.word_vectors.200d.txt.bz2
```

- Start Elasticsearch and check health

```
$ docker-compose up
...

$ curl http://127.0.0.1:9200/_cat/health
1712713293 01:41:33 docker-cluster green 1 1 0 0 0 0 0 0 - 100.0%
```

- Create venv and install packages, and build index

```
$ ~/.pyenv/versions/3.11.8/bin/python -m venv .venv --prompt swem
$ source .venv/bin/activate
$ pip install -r requirements.txt

$ python build_index_wikipedia.py
```

## Text Similarity Search

```
$ python search.py
```
