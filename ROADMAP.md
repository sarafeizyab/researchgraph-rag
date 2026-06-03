# ResearchGraph-RAG Roadmap

## Aim
Build a production-grade, multi-hop scientific RAG platform that provides citation-grounded answers, measurable quality, and deployable API infrastructure.

## Process Overview
1. Ingest and normalize scientific documents from multiple sources.
2. Create metadata-rich chunks and index them in dense + sparse retrievers.
3. Run hybrid retrieval and reranking for high-precision context.
4. Use multi-hop agentic reasoning to gather missing evidence.
5. Synthesize citation-backed answers with transparent reasoning traces.
6. Evaluate quality and compare system variants with ablations.
7. Ship with API, Docker, tests, and CI for public reproducibility.

## Phases

### Phase 1: Foundation (Complete)
- Project scaffolding, config management, logging
- Docker + docker-compose for API and Qdrant
- Core modules and test harness

### Phase 2: Retrieval Stack (Complete)
- PDF/DOCX/URL/arXiv ingestion
- Chunking with overlap and metadata preservation
- Embedding wrapper (OpenAI/local)
- Qdrant dense retrieval + BM25 sparse retrieval
- Hybrid fusion (RRF) + cross-encoder reranking

### Phase 3: Agentic QA (Complete)
- Query decomposition
- Multi-hop retrieve/reflect loop
- Context sufficiency checks and follow-up query generation
- Citation-grounded synthesis

### Phase 4: API and Streaming (Complete)
- `/health`, `/ingest`, `/query`, `/query/stream`
- Pydantic schemas
- SSE event output for reasoning trace and final response

### Phase 5: Evaluation and Ablations (Complete baseline)
- Synthetic testset generation
- Custom fallback metrics
- Ablation summary tooling (JSON + Markdown)

### Phase 6: Public-Repo Hardening (Complete baseline)
- Unit tests
- GitHub Actions CI
- README architecture and design rationale

## Recommended Next Enhancements
1. Add claim-level NLI verification for unsupported-claim detection.
2. Add full RAGAs integration with controlled evaluation datasets.
3. Add MLflow/OpenTelemetry tracing for per-node observability.
4. Add benchmark datasets and frozen evaluation snapshots for regression testing.
