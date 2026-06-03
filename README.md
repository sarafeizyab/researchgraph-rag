# ResearchGraph-RAG: Multi-Hop Scientific RAG with Evaluation

ResearchGraph-RAG is a production-style retrieval-augmented generation system for **complex scientific multi-hop QA**, not a simple single-pass PDF chatbot.

It combines:
- multi-source ingestion (PDF, DOCX, URL, arXiv)
- hybrid retrieval (dense Qdrant + sparse BM25)
- persisted sparse index state for API restarts
- reciprocal-rank fusion and cross-encoder reranking
- multi-hop agentic reasoning with query decomposition and self-reflection
- citation-grounded synthesis
- API + live step streaming + evaluation + ablations

## Why This Is Not a Simple Chatbot

A basic chatbot retrieves once and answers once. This system:
- decomposes hard questions into sub-queries
- iteratively retrieves and reflects across hops
- re-queries when evidence is insufficient
- enforces citation-backed factual claims
- logs latency and token usage
- supports evaluation and ablation for scientific rigor

## Architecture

```mermaid
flowchart TD
    A[Ingestion: PDF/DOCX/URL/arXiv] --> B[Chunker 512/50 overlap]
    B --> C[Embeddings OpenAI or Local]
    C --> D[Qdrant Dense Index]
    B --> E[BM25 Sparse Index]

    Q[User Question] --> G[Decomposer]
    G --> H[Retriever Node]
    H --> D
    H --> E
    D --> I[RRF Fusion]
    E --> I
    I --> J[Cross-Encoder Reranker]
    J --> K[Context Accumulator]
    K --> L[Self-Reflector]
    L -->|Insufficient| H
    L -->|Sufficient or Max Hops| M[Synthesizer]
    M --> N[Answer + Citations + Trace + Metrics]

    N --> O[FastAPI /query]
    N --> P[FastAPI /query/stream SSE]
```

## Repository Structure

```text
researchgraph-rag/
├── ingestion/
├── retrieval/
├── agent/
├── api/
├── evaluation/
├── scripts/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── config.py
├── README.md
└── Makefile
```

## Setup

1. Create environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy configuration:
```bash
cp .env.example .env
```

3. Start Qdrant + API:
```bash
docker compose up --build
```

Or run API locally (with Qdrant already running):
```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` in a browser for the interactive API UI. The root URL
`http://localhost:8000` returns a compact service summary.

## LLM Providers

ResearchGraph-RAG supports three LLM backends:

| Provider | Best for | Cost profile |
|---|---|---|
| OpenAI | strongest default quality and simplest setup | paid API |
| Hugging Face Dedicated Endpoint | managed open-model deployment | paid endpoint, scale-to-zero capable |
| Ollama | local/free development testing | local machine resources |

### OpenAI

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### Hugging Face Dedicated Inference Endpoint

Create a dedicated text-generation endpoint in Hugging Face, copy the endpoint URL, and set:

```bash
LLM_PROVIDER=huggingface
HF_TOKEN=your_huggingface_token
HF_ENDPOINT_URL=https://your-endpoint.endpoints.huggingface.cloud
HF_ENDPOINT_MODE=text-generation
HF_MODEL=Qwen/Qwen2.5-7B-Instruct
HF_MAX_NEW_TOKENS=768
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

If the endpoint exposes OpenAI-compatible chat completions, use:

```bash
HF_ENDPOINT_MODE=chat-completions
HF_ENDPOINT_URL=https://your-endpoint.endpoints.huggingface.cloud
```

### Ollama

Install Ollama and pull a model:

```bash
ollama pull llama3.1:8b
```

Then configure:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

For local non-Docker runs, use `OLLAMA_BASE_URL=http://localhost:11434`.

### Future-Compatible TGI Path

Self-hosted Hugging Face Text Generation Inference can be added later as another provider. It is operationally heavier than a dedicated endpoint, but the provider interface is already shaped to support a configurable model server.

## Troubleshooting

If Docker Compose returns a Qdrant connection error, rebuild after pulling the latest config:
```bash
docker compose up --build
```

Inside Docker, the API uses `QDRANT_URL=http://qdrant:6333`. For local non-Docker runs, use `QDRANT_URL=http://localhost:6333`.

If `/query` returns an OpenAI quota or billing error, the API key is valid enough to call OpenAI but has no available quota. Use a key with billing enabled, or switch embeddings to a local model to reduce OpenAI usage:
```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Ingest Documents

### API ingestion options
- PDF upload
- DOCX upload
- URL
- arXiv ID

Example:
```bash
curl -X POST http://localhost:8000/ingest -F "url=https://arxiv.org/abs/1706.03762"
```

## Ask Questions

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare transformer-based and CNN-based models for retinal disease classification and discuss explainability trade-offs."}'
```

### Streaming reasoning
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"How do foundation models compare with task-specific genomics models for variant interpretation?"}'
```

The streaming endpoint emits `start`, `reasoning_step`, `final_answer`, and `done` events as the agent progresses through decomposition, retrieval, reflection, and synthesis.

## Example Multi-Hop Output (shape)

```json
{
  "answer": "Transformers improved long-range feature modeling in retinal OCT studies [3f4c...], while CNN baselines remained competitive on smaller datasets [91a2...].",
  "citations": [
    {
      "chunk_id": "3f4c...",
      "source_doc": "retina_paper.pdf",
      "page": 5,
      "section": "Results",
      "excerpt": "..."
    }
  ],
  "reasoning_trace": [
    {"step":"decompose","sub_queries":["...", "..."]},
    {"step":"retrieve","query":"...","retrieved":5,"new_chunks_added":4},
    {"step":"reflect","hop":1,"sufficient":false,"follow_up_query":"..."},
    {"step":"synthesize","cited_chunks":["..."]}
  ]
}
```

## Evaluation Methodology

`evaluation/` includes:
- synthetic test-set generation from ingested chunks
- fallback custom metrics:
  - citation coverage
  - retrieved context overlap
  - answer length
  - cited claim count
  - unsupported-claim detector placeholder

If RAGAs is available in your environment, you can extend `evaluation/metrics.py` with faithfulness/relevancy/context metrics.

## Ablation Experiments

Compare:
- dense-only vs hybrid retrieval
- hybrid without reranker vs hybrid with reranker
- 1-hop vs multi-hop

Export:
- JSON summary
- Markdown table for README/reports

### Placeholder table

| Variant | Metric | Value |
|---|---:|---:|
| dense_only | avg_citation_coverage | TBD |
| hybrid_no_rerank | avg_citation_coverage | TBD |
| hybrid_rerank | avg_citation_coverage | TBD |
| multi_hop | avg_citation_coverage | TBD |

## Design Decisions

- **LangGraph**: explicit state transitions, inspectable reasoning graph, controlled multi-hop behavior.
- **Hybrid retrieval**: dense semantic recall + BM25 lexical precision reduces blind spots.
- **Reranking**: cross-encoder boosts top-context relevance before synthesis.
- **Self-reflection**: avoids premature synthesis when evidence is weak.
- **Evaluation-first framing**: supports reproducibility and engineering credibility.

## Tests

Run:
```bash
pytest -q
```

Current tests cover:
- chunking behavior
- BM25 retrieval
- RRF hybrid fusion
- reranking wrapper
- citation metrics
- FastAPI health endpoint
- agent state transitions

## Future Improvements

- introduce explicit claim-level verifier (NLI) for unsupported claim detection
- add optional graph-of-papers citation traversal
- add MLflow tracing for retrieval/hop-level diagnostics
- add authenticated multi-tenant document collections
- add CI pipeline for benchmark regression checks
