# Retrieval‑Augmented Generation (RAG) Playground

A minimal, end‑to‑end demo that evolves from **classic keyword search** to **vector‑based semantic retrieval** while keeping the rest of the RAG pipeline constant. Perfect for learning how each retrieval style impacts answer quality, latency, and cost.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Diagram](#pipeline-diagram)
3. [Quick Start](#quick-start)
4. [Dataset](#dataset)
5. [Retrieval Strategies](#retrieval-strategies)
6. [Project Structure](#project-structure)
7. [Roadmap](#roadmap)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

Retrieval‑Augmented Generation (RAG) grounds an LLM’s answers in an external knowledge base. This repo shows three levels of retrieval in a single code‑base:

| Level | Technique         | Library              | One‑liner                          |
| ----- | ----------------- | -------------------- | ---------------------------------- |
| 1     | **TF‑IDF / BM25** | `scikit‑learn`       | Classic bag‑of‑words baseline      |
| 2     | **Elasticsearch** | `elasticsearch-py`   | Production inverted index, filters |
| 3     | **Dense vectors** | `faiss` / `pgvector` | Semantic recall with ANN           |

At every level we reuse the **same dataset** and **evaluation harness** so the comparison is apples‑to‑apples.

---

## Pipeline Diagram

Add (or regenerate) the architecture image in `docs/img/rag_pipeline.svg` and reference it like this:

```markdown
![RAG pipeline schematic](docs/img/rag_pipeline.svg)
```

<details>
<summary>Legend</summary>

1. **Indexing** → build/refresh search index
2. **Retrieval** → fetch top‑k passages for the user query
3. **Augmentation** → compose prompt: `<context> + <question>`
4. **Generation** → call LLM (OpenAI, Ollama, etc.)
5. **Post‑processing** → optional re‑ranking / citation extraction

</details>

---

## Quick Start

```bash
# clone & install
$ git clone https://github.com/<your‑org>/rag-playground.git
$ cd rag-playground
$ pip install -r requirements.txt

# run end‑to‑end with TF‑IDF
$ python pipelines/tfidf_demo.py --query "How do I run Kafka?"
```

Switch retrieval style with `--retriever {tfidf,elastic,dense}`.

---

## Dataset

A trimmed StackOverflow dump (\~10 k docs) lives in `data/stackoverflow.csv`. Swap in your own corpus by dropping files into `data/` and updating `pipelines/common.py`.

---

## Retrieval Strategies

| Flag      | Description            | Index                  | Pros               | Cons            |
| --------- | ---------------------- | ---------------------- | ------------------ | --------------- |
| `tfidf`   | Sparse TF‑IDF matrix   | in‑memory              | fast, no infra     | lexical only    |
| `elastic` | BM25 in Elasticsearch  | on‑disk inverted index | filters, fuzziness | Java, memory    |
| `dense`   | Cosine over embeddings | FAISS / pgvector       | semantic recall    | embeddings cost |

---

## Project Structure

```
rag-playground/
├── data/                # raw & processed corpora
├── docs/img/            # architecture diagrams & figures
├── pipelines/           # tfidf_demo.py, elastic_demo.py, dense_demo.py
├── evaluation/          # exact‑match, Rouge‑L, latency scripts
├── notebooks/           # exploratory analyses, metric plots
├── requirements.txt     # minimal deps
└── README.md            # you are here
```

---

## Roadmap

* [x] TF‑IDF baseline
* [ ] Dockerised Elasticsearch + BM25
* [ ] GPU FAISS index + ANN
* [ ] LangChain runnable graph
* [ ] Streamlit chat UI with citations
* [ ] Formal evaluation harness

---

## Contributing

PRs, issues, and feature requests are welcome! Please see `CONTRIBUTING.md` for guidelines.

---

## License

MIT © 2025 Rishav & Contributors
